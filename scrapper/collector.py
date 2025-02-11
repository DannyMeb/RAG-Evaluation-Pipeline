# Last update: Today
# Project: Rag systems evaluation pipeline using Ragas
# Authored by TII/AICCU/Edge-Team
# Final version: Implemented multi-system support (RagFlow, Dify) and improved API handling.


import os
import re
import argparse
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PyPDF2 import PdfMerger
from tqdm import tqdm  # For progress bar
import matplotlib.pyplot as plt
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Utility Functions

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', filename)

def sanitize_text(text):
    if text and isinstance(text, str):
        soup = BeautifulSoup(text, 'html.parser')
        clean_text = soup.get_text()
        # Remove common boilerplate text
        boilerplate_keywords = ["Subscribe", "copyright", "cookies", "advertisement"]
        for keyword in boilerplate_keywords:
            clean_text = clean_text.replace(keyword, '')
        return clean_text.strip()
    return str(text)

def scrape_article_content(url, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                paragraphs = soup.find_all('p')
                content = ' '.join([p.text for p in paragraphs if p.text])  # Ensure paragraphs have text
                return content if content else "No content available"
            else:
                print(f"Failed to retrieve content from {url}, status code: {response.status_code}")
        except requests.Timeout:
            print(f"Request timed out for {url}, retrying ({attempt + 1}/{retries})...")
            time.sleep(2)  # Delay before retrying
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
    return "Scraping failed"

def filter_article_content(content):
    word_count = len(content.split())
    # Discard articles with less than 100 words or no meaningful content
    if word_count < 100 or "No content available" in content:
        return False
    return True

def add_article_to_pdf(merger, title, description, content, published_at, author, source_name, url):
    from io import BytesIO
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=4))

    story = []
    title = sanitize_text(title)
    description = sanitize_text(description)
    content = sanitize_text(content)
    author = sanitize_text(author)
    source_name = sanitize_text(source_name)
    
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Author: {author}", styles['Italic']))
    story.append(Paragraph(f"Source: {source_name}", styles['Italic']))
    story.append(Paragraph(f"Published: {published_at}", styles['Italic']))
    story.append(Paragraph(f"URL: {url}", styles['Italic']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("Description:", styles['Heading2']))
    story.append(Paragraph(description, styles['Justify']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("Full Content:", styles['Heading2']))
    story.append(Paragraph(content, styles['Justify']))

    doc.build(story)
    
    buffer.seek(0)
    merger.append(buffer)

def NewsFromTopic(newsapi, topic, articles_per_topic, merger):
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    total_fetched_articles = 0
    page = 1
    while total_fetched_articles < articles_per_topic:
        remaining_articles = articles_per_topic - total_fetched_articles
        page_size = min(remaining_articles, 100)  # Limit to 100 articles per page

        all_articles = newsapi.get_everything(q=topic,
                                              from_param=from_date,
                                              to=to_date,
                                              language='en',
                                              sort_by='publishedAt',
                                              page=page,
                                              page_size=page_size)

        if not all_articles['articles']:
            break

        for article in tqdm(all_articles['articles'], desc=f"Processing {topic}", unit="article", total=page_size):
            try:
                title = article.get("title", "No Title")
                description = article.get("description", "No Description")
                published_at = article.get("publishedAt", "No Date")
                author = article.get("author", "No Author")
                source_name = article.get("source", {}).get("name", "No Source")
                url = article.get("url", "No URL")

                full_content = scrape_article_content(url)
                
                # Only add to PDF if content passes filter
                if filter_article_content(full_content):
                    add_article_to_pdf(merger, title, description, full_content, published_at, author, source_name, url)

            except Exception as e:
                print(f"Error processing article: {str(e)}")

        total_fetched_articles += len(all_articles['articles'])
        page += 1

    return total_fetched_articles

def remove_duplicates(articles):
    unique_articles = []
    tfidf = TfidfVectorizer().fit_transform([article['content'] for article in articles])
    pairwise_similarities = cosine_similarity(tfidf, tfidf)
    
    for i, article in enumerate(articles):
        if all(pairwise_similarities[i][j] < 0.9 for j in range(i)):  # Similarity threshold of 0.9
            unique_articles.append(article)
    return unique_articles

# PDF Merging Functions

def create_title_page(merger, category):
    from io import BytesIO
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    story = []
    story.append(Paragraph(f"{category} News Articles", styles['Title']))
    story.append(PageBreak())

    doc.build(story)

    buffer.seek(0)
    merger.append(buffer)

# Function to generate pie chart
def create_pie_chart(topic_article_counts, output_dir):
    labels = list(topic_article_counts.keys())
    sizes = list(topic_article_counts.values())
    total_articles = sum(sizes)  # Calculate the total number of articles

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Add total articles count in the bottom right corner
    plt.text(0.5, -1.2, f'Total Articles: {total_articles}', ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Save pie chart in the output directory
    pie_chart_path = os.path.join(output_dir, "topic_distribution_pie_chart.png")
    plt.savefig(pie_chart_path)
    plt.close()

    print(f"Pie chart saved at: {pie_chart_path}")
    return pie_chart_path

# Main Function

def main(articles_per_topic):
    # Start time
    tik = time.time()
    base_dir = "/home/ubuntu/ragas/data"
    final_output_path = os.path.join(base_dir, "knowledge_graph_source.pdf")

    os.makedirs(base_dir, exist_ok=True)

    API_KEY = "01287c60979f4f2db0c6b699b68983e3"
    newsapi = NewsApiClient(api_key=API_KEY)
    
    topics = ["Agriculture", "Artificial Intelligence", "Arts", "Architecture", "Aviation", "Automotive", 
              "Business", "Climate Change", "Economy", "Education", "Entertainment", "Health", 
              "Human Rights","Law", "Military", "Politics", "Religion", "Science", "Space Exploration", 
              "Sports", "Technology", "Transportation"]

    final_merger = PdfMerger()

    topic_article_counts = {}

    for topic in tqdm(topics, desc="Processing topics", unit="topic"):
        create_title_page(final_merger, topic)
        article_count = NewsFromTopic(newsapi, topic, articles_per_topic, final_merger)
        topic_article_counts[topic] = article_count  # Track the number of articles for each topic

    with open(final_output_path, 'wb') as f:
        final_merger.write(f)

    final_merger.close()

    print(f"Final merged PDF created: {final_output_path}")

    # Generate pie chart for topic distributions
    create_pie_chart(topic_article_counts, base_dir)

    # End time
    tok = time.time()

    # Calculate total time
    total_time = tok - tik
    print(f"\nPipeline completed successfully in {total_time:.2f} seconds!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download news articles, merge them into a single PDF.")
    parser.add_argument('--articles_per_topic', type=int, default=5, help='Number of articles to download per topic (default: 5)')
    args = parser.parse_args()

    main(args.articles_per_topic)
