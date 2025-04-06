import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import nltk # type: ignore
from nltk.corpus import stopwords # type: ignore
import requests
from bs4 import BeautifulSoup
import spacy
import json
import time
import logging
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kukufm_recommender.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("KukuFM_Recommender")

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class KukuFMRecommendationSystem:
    def __init__(self, book_catalog_path, news_sources=None, language='english', region='india'):
        """
        Initialize the recommendation system with book catalog and news sources
        
        Parameters:
        book_catalog_path (str): Path to the CSV file containing book information
        news_sources (list): List of URLs to scrape for current news
        language (str): Language for stopwords and analysis
        region (str): Country/region for localizing news sources
        """
        self.region = region
        self.language = language
        
        # Set language-specific resources
        self.setup_language_resources()
        
        # Load book catalog
        logger.info(f"Loading book catalog from {book_catalog_path}")
        self.book_df = pd.read_csv(book_catalog_path)
        self.clean_book_data()
        
        # Set news sources
        self.news_sources = news_sources or self.get_default_news_sources()
        logger.info(f"Using {len(self.news_sources)} news sources for {self.region}")
        
        # Initialize trending topics
        self.trending_topics = []
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words=self.stop_words_list,
            ngram_range=(1, 2),
            max_features=5000
        )
        
        # Process book data with TF-IDF
        logger.info("Processing book data with TF-IDF")
        self.book_tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.book_df['processed_text']
        )
        
        # Cache for scraped news
        self.news_cache = {'timestamp': None, 'articles': []}
        self.cache_expiry = 3600  # Cache news for 1 hour
    
    def setup_language_resources(self):
        """Set up language-specific resources for text processing"""
        try:
            if self.language == 'english':
                self.stop_words_list = list(stopwords.words('english'))
                self.nlp = spacy.load('en_core_web_md')
            elif self.language == 'hindi':
                self.stop_words_list = list(stopwords.words('hindi')) if 'hindi' in stopwords._fileids else []
                self.nlp = spacy.load('xx_ent_wiki_sm')  # Alternative if hindi model not available
            else:
                # Default to English if language not supported
                logger.warning(f"Language {self.language} not fully supported, using English resources")
                self.stop_words_list = list(stopwords.words('english'))
                self.nlp = spacy.load('en_core_web_md')
        except OSError:
            logger.warning(f"Language model not available, using small model instead")
            self.nlp = spacy.load('en_core_web_sm')
    
    def get_default_news_sources(self):
        """Get default news sources based on region"""
        news_sources = {
            'india': [
                "https://timesofindia.indiatimes.com/",
                "https://www.ndtv.com/",
                "https://www.hindustantimes.com/",
                "https://economictimes.indiatimes.com/",
                "https://indianexpress.com/"
            ],
            'usa': [
                "https://www.cnn.com/",
                "https://www.nytimes.com/",
                "https://www.washingtonpost.com/",
                "https://www.foxnews.com/",
                "https://www.usatoday.com/"
            ]
        }
        
        return news_sources.get(self.region.lower(), news_sources['india'])
    
    def clean_book_data(self):
        """Clean and prepare book data"""
        logger.info("Cleaning and preparing book data")
        
        # Handle missing values
        for col in ['description', 'keywords', 'genres']:
            if col in self.book_df.columns:
                self.book_df[col] = self.book_df[col].fillna('')
        
        # Ensure required columns exist
        required_cols = ['title', 'author', 'description', 'keywords', 'genres']
        for col in required_cols:
            if col not in self.book_df.columns:
                self.book_df[col] = ''
                logger.warning(f"Created empty column '{col}' in book data")
        
        # Create processed text column for TF-IDF
        self.book_df['processed_text'] = (
            self.book_df['title'] + ' ' + 
            self.book_df['author'] + ' ' + 
            self.book_df['description'] + ' ' + 
            self.book_df['keywords'] + ' ' + 
            self.book_df['genres']
        )
    
    def scrape_current_news(self, force_refresh=False):
        """
        Scrape current news from the specified sources
        
        Parameters:
        force_refresh (bool): Force refresh even if cache is valid
        
        Returns:
        list: List of article dictionaries with headline, content, and source
        """
        current_time = time.time()
        
        # Use cached news if available and not expired
        if (not force_refresh and 
            self.news_cache['timestamp'] is not None and 
            current_time - self.news_cache['timestamp'] < self.cache_expiry):
            logger.info("Using cached news data")
            return self.news_cache['articles']
        
        logger.info("Scraping news from sources")
        all_articles = []
        
        for source in self.news_sources:
            try:
                logger.info(f"Scraping news from {source}")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(source, headers=headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract headlines and article text
                # Note: These selectors need to be adjusted for each specific news source
                headlines = soup.select('h1, h2, h3')[:10]  # Limit to top 10 headlines
                
                for headline in headlines:
                    # Extract headline text
                    headline_text = headline.text.strip()
                    
                    # Skip very short headlines
                    if len(headline_text) < 10:
                        continue
                    
                    # Try to find related content
                    parent = headline.parent
                    content = ""
                    
                    # Look for paragraph text near the headline
                    paragraphs = parent.select('p')
                    if paragraphs:
                        content = ' '.join([p.text.strip() for p in paragraphs[:3]])
                    
                    # If no content found, use the headline as content
                    if not content:
                        content = headline_text
                    
                    all_articles.append({
                        'headline': headline_text,
                        'content': content,
                        'source': source
                    })
                
                # Respect the website's crawl rate
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error scraping {source}: {e}")
        
        # Update cache
        self.news_cache = {
            'timestamp': current_time,
            'articles': all_articles
        }
        
        logger.info(f"Scraped {len(all_articles)} articles from {len(self.news_sources)} sources")
        return all_articles
    
    def extract_trending_topics(self, articles, num_topics=10):
        """
        Extract trending topics from scraped news articles
        
        Parameters:
        articles (list): List of article dictionaries
        num_topics (int): Number of top topics to extract
        
        Returns:
        list: List of (topic, count) tuples
        """
        if not articles:
            logger.warning("No articles to extract topics from")
            return []
        
        logger.info(f"Extracting trending topics from {len(articles)} articles")
        
        # Combine article headlines and content
        all_content = ' '.join([f"{a['headline']} {a['content']}" for a in articles])
        
        # Process with spaCy
        doc = self.nlp(all_content[:1000000])  # Limit text size to avoid memory issues
        
        # Extract named entities
        entities = {}
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 'NORP', 'FAC']:
                # Normalize entity text
                entity_text = ent.text.strip()
                if len(entity_text) > 3:  # Skip very short entities
                    entities[entity_text] = entities.get(entity_text, 0) + 1
        
        # Extract noun phrases
        noun_phrases = {}
        for chunk in doc.noun_chunks:
            # Clean and normalize the noun phrase
            phrase = chunk.text.strip()
            words = phrase.split()
            
            # Filter out short phrases and those containing only stopwords
            if (len(words) > 1 and len(phrase) > 5 and 
                not all(word.lower() in self.stop_words_list for word in words)):
                noun_phrases[phrase] = noun_phrases.get(phrase, 0) + 1
        
        # Extract keywords using TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        article_texts = [f"{a['headline']} {a['content']}" for a in articles]
        
        tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words=self.stop_words_list,
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(article_texts)
            feature_names = tfidf_vectorizer.get_feature_names_out()
            
            # Get top keywords across all articles
            tfidf_scores = tfidf_matrix.sum(axis=0).A1
            top_indices = tfidf_scores.argsort()[-50:][::-1]
            
            keywords = {feature_names[i]: int(tfidf_scores[i] * 10) for i in top_indices}
        except Exception as e:
            logger.error(f"Error extracting TF-IDF keywords: {e}")
            keywords = {}
        
        # Combine all topic sources with weights
        topics = {}
        for source, weight in [(entities, 3), (noun_phrases, 2), (keywords, 1)]:
            for topic, count in source.items():
                topics[topic] = topics.get(topic, 0) + (count * weight)
        
        # Sort by count (frequency)
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
        
        # Remove duplicates and near-duplicates
        filtered_topics = []
        added_topics_lower = set()
        
        for topic, count in sorted_topics:
            # Skip if too similar to already added topics
            topic_lower = topic.lower()
            if any(topic_lower in added.lower() or added.lower() in topic_lower for added in added_topics_lower):
                continue
            
            filtered_topics.append((topic, count))
            added_topics_lower.add(topic_lower)
            
            # Stop once we have enough topics
            if len(filtered_topics) >= num_topics:
                break
        
        self.trending_topics = filtered_topics
        logger.info(f"Extracted {len(self.trending_topics)} trending topics")
        
        return self.trending_topics
    
    def find_relevant_books(self, topics, num_recommendations=5):
        """
        Find books relevant to the trending topics
        
        Parameters:
        topics (list): List of (topic, count) tuples
        num_recommendations (int): Number of book recommendations to return
        
        Returns:
        DataFrame: DataFrame of recommended books with relevance scores
        """
        if not topics:
            logger.warning("No topics to find relevant books for")
            return pd.DataFrame()
        
        logger.info(f"Finding books relevant to {len(topics)} topics")
        
        # Create a query from trending topics, weighted by topic frequency
        query_parts = []
        for topic, count in topics:
            # Add the topic multiple times based on its count
            weight = min(count, 5)  # Cap at 5 to avoid overweighting
            query_parts.extend([topic] * weight)
        
        query = ' '.join(query_parts)
        
        # Transform query using the same vectorizer
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarity with all books
        similarity_scores = cosine_similarity(query_vector, self.book_tfidf_matrix).flatten()
        
        # Get indices of top recommendations
        top_indices = similarity_scores.argsort()[-num_recommendations*2:][::-1]  # Get extra for filtering
        
        # Return recommended books
        recommendations = self.book_df.iloc[top_indices].copy()
        recommendations['relevance_score'] = similarity_scores[top_indices]
        
        # Filter out books with very low relevance
        recommendations = recommendations[recommendations['relevance_score'] > 0.1]
        
        # Limit to requested number
        recommendations = recommendations.head(num_recommendations)
        
        logger.info(f"Found {len(recommendations)} relevant books")
        return recommendations
    
    def generate_recommendations(self, num_recommendations=5, force_refresh=False):
        """
        Generate book recommendations based on current news
        
        Parameters:
        num_recommendations (int): Number of book recommendations to return
        force_refresh (bool): Force refresh of news data
        
        Returns:
        tuple: (DataFrame of recommendations, list of trending topics)
        """
        logger.info("Generating recommendations based on current news")
        
        # Scrape current news
        articles = self.scrape_current_news(force_refresh=force_refresh)
        
        # Extract trending topics
        topics = self.extract_trending_topics(articles)
        
        if not topics:
            logger.warning("No trending topics found")
            return pd.DataFrame(), []
        
        # Find relevant books
        recommendations = self.find_relevant_books(topics, num_recommendations)
        
        if recommendations.empty:
            logger.warning("No relevant books found")
            return recommendations, topics
        
        # Map each book to the topics it matches
        for idx, book in recommendations.iterrows():
            book_text = book['processed_text'].lower()
            matching_topics = []
            
            for topic, _ in topics:
                if topic.lower() in book_text:
                    matching_topics.append(topic)
            
            recommendations.at[idx, 'matching_topics'] = ', '.join(matching_topics[:5])
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations, topics
    
    def personalize_recommendations(self, user_id, base_recommendations, user_preferences):
        """
        Personalize recommendations for a specific user
        
        Parameters:
        user_id (int): User identifier
        base_recommendations (DataFrame): Base recommendations from trending topics
        user_preferences (DataFrame): User preference information
        
        Returns:
        DataFrame: Personalized recommendations
        """
        if base_recommendations.empty:
            logger.warning("No base recommendations to personalize")
            return base_recommendations
        
        logger.info(f"Personalizing recommendations for user {user_id}")
        
        # Get user reading history and preferences
        user_pref = user_preferences[user_preferences['user_id'] == user_id]
        
        if user_pref.empty:
            logger.info(f"No preference data for user {user_id}, using base recommendations")
            return base_recommendations
        
        # Create a copy to modify
        personalized = base_recommendations.copy()
        
        try:
            # Adjust scores based on user genre preferences
            preferred_genres = user_pref['preferred_genres'].iloc[0].split(',')
            
            # Boost scores for preferred genres
            for idx, book in personalized.iterrows():
                book_genres = str(book['genres']).split(',')
                genre_match = sum(genre.strip().lower() in [pg.strip().lower() for pg in preferred_genres] for genre in book_genres)
                boost = 0.2 * genre_match  # 20% boost per matching genre
                
                personalized.at[idx, 'relevance_score'] = book['relevance_score'] * (1 + boost)
            
            # Adjust based on user's reading history if available
            if 'reading_history' in user_pref.columns:
                reading_history = user_pref['reading_history'].iloc[0].split(',')
                
                # Boost books by authors the user has read before
                for idx, book in personalized.iterrows():
                    if any(author.strip().lower() in book['author'].lower() for author in reading_history):
                        personalized.at[idx, 'relevance_score'] *= 1.3  # 30% boost
            
            # Re-sort by adjusted scores
            personalized = personalized.sort_values('relevance_score', ascending=False)
            
            logger.info(f"Successfully personalized recommendations for user {user_id}")
        except Exception as e:
            logger.error(f"Error personalizing recommendations: {e}")
        
        return personalized
    
    def get_trending_news_summary(self, articles, num_articles=5):
        """
        Generate a summary of trending news articles
        
        Parameters:
        articles (list): List of article dictionaries
        num_articles (int): Number of articles to include in summary
        
        Returns:
        list: List of summarized article dictionaries
        """
        if not articles:
            return []
        
        # Sort articles by content length as a simple heuristic
        sorted_articles = sorted(articles, key=lambda x: len(x['content']), reverse=True)
        
        # Take top articles
        top_articles = sorted_articles[:num_articles]
        
        # Generate summaries
        summaries = []
        for article in top_articles:
            # Use basic extractive summarization
            doc = self.nlp(article['content'])
            sentences = [sent.text.strip() for sent in doc.sents]
            
            # Take first 2 sentences or headline if content is short
            if len(sentences) >= 2:
                summary = ' '.join(sentences[:2])
            else:
                summary = article['headline']
            
            summaries.append({
                'headline': article['headline'],
                'summary': summary,
                'source': article['source']
            })
        
        return summaries
    
    def export_recommendations(self, recommendations, topics, news_summaries, output_file=None):
        """
        Export recommendations to JSON format
        
        Parameters:
        recommendations (DataFrame): Recommended books
        topics (list): List of trending topics
        news_summaries (list): List of news article summaries
        output_file (str): Path to output file, if None returns JSON string
        
        Returns:
        str: JSON string if output_file is None
        """
        # Prepare recommendations output
        rec_list = []
        for _, book in recommendations.iterrows():
            rec_list.append({
                'title': book['title'],
                'author': book['author'],
                'relevance_score': float(book['relevance_score']),
                'matching_topics': book['matching_topics'].split(', ') if book['matching_topics'] else [],
                'genres': book['genres'].split(',') if isinstance(book['genres'], str) else []
            })
        
        # Prepare topics output
        topic_list = [{'topic': topic, 'frequency': count} for topic, count in topics]
        
        # Create output dictionary
        output = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'trending_topics': topic_list,
            'news_summaries': news_summaries,
            'recommended_books': rec_list
        }
        
        # Write to file or return as string
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            return f"Recommendations exported to {output_file}"
        else:
            return json.dumps(output, indent=2, ensure_ascii=False)

def create_sample_data():
    """Create sample data files for testing"""
    # Create sample book catalog
    sample_books = pd.DataFrame({
        'title': [
            'The Art of War', 'Democracy in America', 'The Climate Crisis',
            'Digital Minimalism', 'Economic Principles', 'Health Revolution',
            'AI and the Future of Work', 'The History of India', 'Global Diplomacy',
            'Modern Politics'
        ],
        'author': [
            'Sun Tzu', 'Alexis de Tocqueville', 'James Hansen',
            'Cal Newport', 'N. Gregory Mankiw', 'Dr. Michael Greger',
            'Kai-Fu Lee', 'Romila Thapar', 'Henry Kissinger',
            'Political Analyst'
        ],
        'description': [
            'Ancient Chinese military treatise dating from the 5th century BC.',
            'Analysis of the United States and its political system.',
            'An exploration of climate change effects and solutions.',
            'A philosophy of technology use in a distracted world.',
            'Fundamental principles of economics explained.',
            'How nutrition affects health and longevity.',
            'Exploring how artificial intelligence will transform employment.',
            'Comprehensive history of the Indian subcontinent.',
            'International relations and diplomacy in the modern world.',
            'Analysis of contemporary political systems.'
        ],
        'keywords': [
            'strategy, warfare, leadership, military',
            'democracy, politics, america, government',
            'environment, climate, global warming, sustainability',
            'technology, focus, productivity, internet',
            'economics, market, finance, money',
            'health, nutrition, diet, medicine',
            'artificial intelligence, technology, future, jobs',
            'india, history, culture, civilization',
            'international relations, politics, diplomacy, conflict',
            'politics, governance, democracy, power'
        ],
        'genres': [
            'Philosophy,History',
            'Politics,History',
            'Science,Current Affairs',
            'Technology,Self-Help',
            'Business,Education',
            'Health,Science',
            'Technology,Business',
            'History,Culture',
            'Politics,History',
            'Politics,Current Affairs'
        ]
    })
    
    sample_books.to_csv('sample_books.csv', index=False)
    
    # Create sample user preferences
    sample_users = pd.DataFrame({
        'user_id': [1, 2, 3],
        'preferred_genres': [
            'Politics,History,Current Affairs',
            'Technology,Science,Business',
            'Health,Self-Help,Culture'
        ],
        'reading_history': [
            'Henry Kissinger,Political Analyst',
            'Kai-Fu Lee,Cal Newport',
            'Dr. Michael Greger,Romila Thapar'
        ]
    })
    
    sample_users.to_csv('sample_users.csv', index=False)
    
    return 'sample_books.csv', 'sample_users.csv'

def main():
    parser = argparse.ArgumentParser(description='KukuFM Smart Content Curation System')
    parser.add_argument('--books', type=str, help='Path to book catalog CSV file')
    parser.add_argument('--users', type=str, help='Path to user preferences CSV file')
    parser.add_argument('--region', type=str, default='india', help='Region for news sources')
    parser.add_argument('--language', type=str, default='english', help='Language for processing')
    parser.add_argument('--recommendations', type=int, default=5, help='Number of recommendations to generate')
    parser.add_argument('--user_id', type=int, help='User ID for personalized recommendations')
    parser.add_argument('--output', type=str, help='Path to output JSON file')
    parser.add_argument('--create_sample', action='store_true', help='Create sample data files')
    parser.add_argument('--force_refresh', action='store_true', help='Force refresh of news data')
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample:
        books_path, users_path = create_sample_data()
        logger.info(f"Created sample data files: {books_path}, {users_path}")
        if not args.books:
            args.books = books_path
        if not args.users:
            args.users = users_path
    
    # Use sample data if no files provided
    if not args.books:
        try:
            args.books = 'sample_books.csv'
            logger.info(f"Using default book catalog: {args.books}")
        except:
            logger.error("No book catalog file provided or found")
            return
    
    # Initialize the recommendation system
    recommender = KukuFMRecommendationSystem(
        book_catalog_path=args.books,
        language=args.language,
        region=args.region
    )
    
    # Generate recommendations
    recommendations, topics = recommender.generate_recommendations(
        num_recommendations=args.recommendations,
        force_refresh=args.force_refresh
    )
    
    # Get news summaries
    articles = recommender.scrape_current_news()
    news_summaries = recommender.get_trending_news_summary(articles)
    
    # Personalize recommendations if user_id is provided
    if args.user_id and args.users:
        try:
            user_preferences = pd.read_csv(args.users)
            recommendations = recommender.personalize_recommendations(
                args.user_id, recommendations, user_preferences
            )
            logger.info(f"Personalized recommendations for user {args.user_id}")
        except Exception as e:
            logger.error(f"Error personalizing recommendations: {e}")
    
    # Export recommendations
    result = recommender.export_recommendations(
        recommendations, topics, news_summaries, args.output
    )
    
    if args.output:
        logger.info(result)
    else:
        print(result)

if __name__ == "__main__":
    main()