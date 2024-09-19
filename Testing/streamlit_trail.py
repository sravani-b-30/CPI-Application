import pandas as pd
import ast
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import numpy as np
import mplcursors
import nltk
from nltk.tokenize import word_tokenize
from datetime import datetime, timedelta
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import streamlit as st 


nltk.download('punkt')


def format_details(details):
    return "\n".join([f"{key}: {value}" for key, value in details.items()])

def tokenize(text):
    tokens = word_tokenize(text.lower())
    return set(tokens)


def tokenize_with_delimiters(text):
    text = text.lower()
    tokens = re.split(r'[,;.\s]', text)
    return set(token for token in tokens if token)


def extract_numeric_metric(text):
    return set(re.findall(r'\d+\s?[a-zA-Z"]+', text.lower()))


def extract_thickness(dimension_str):
    match = re.search(r'\d+"Th', dimension_str)
    if match:
        return match.group()
    return ''


def tokenized_similarity(value1, value2):
    if value1 is None or value2 is None:
        return 0
    tokens1 = tokenize_with_delimiters(str(value1))
    tokens2 = tokenize_with_delimiters(str(value2))
    numeric_metric1 = extract_numeric_metric(str(value1))
    numeric_metric2 = extract_numeric_metric(str(value2))
    intersection = tokens1.intersection(tokens2).union(numeric_metric1.intersection(numeric_metric2))
    return len(intersection) / len(tokens1.union(tokens2))


def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def title_similarity(title1, title2):
    # Tokenize the titles
    title1_tokens = tokenize_with_delimiters(title1)
    title2_tokens = tokenize_with_delimiters(title2)

    # Calculate intersection and union of title tokens
    intersection = title1_tokens.intersection(title2_tokens)
    union = title1_tokens.union(title2_tokens)

    # Calculate token similarity score
    token_similarity_score = len(intersection) / len(union)

    # Extract numeric metrics
    numeric_metric1 = extract_numeric_metric(title1)
    numeric_metric2 = extract_numeric_metric(title2)

    # Calculate numeric metric match score
    numeric_match_count = len(numeric_metric1.intersection(numeric_metric2))

    # Final similarity score
    similarity_score = (token_similarity_score + numeric_match_count) * 100

    return similarity_score, title1_tokens, title2_tokens, intersection


def description_similarity(desc1, desc2):
    desc1_tokens = tokenize_with_delimiters(desc1)
    desc2_tokens = tokenize_with_delimiters(desc2)
    intersection = desc1_tokens.intersection(desc2_tokens)
    union = desc1_tokens.union(desc2_tokens)
    similarity_score = len(intersection) / len(union) * 100
    return similarity_score, desc1_tokens, desc2_tokens, intersection


def parse_dict_str(dict_str):
    try:
        return ast.literal_eval(dict_str)
    except ValueError:
        return {}


def merge_dicts(dict1, dict2):
    merged = dict1.copy()
    merged.update(dict2)
    return merged


def convert_weight_to_kg(weight_str):
    weight_str = weight_str.lower()
    match = re.search(r'(\d+\.?\d*)\s*(pounds?|lbs?|kg)', weight_str)
    if match:
        value, unit = match.groups()
        value = float(value)
        if 'pound' in unit or 'lb' in unit:
            value *= 0.453592
        return value
    return None


def parse_weight(weight_str):
    weight_kg = convert_weight_to_kg(weight_str)
    return weight_kg


def parse_dimensions(dimension_str):
    matches = re.findall(r'(\d+\.?\d*)\s*"?([a-zA-Z]+)"?', dimension_str)
    if matches:
        return {unit: float(value) for value, unit in matches}
    return {}


def compare_weights(weight1, weight2):
    weight_kg1 = parse_weight(weight1)
    weight_kg2 = parse_weight(weight2)
    if weight_kg1 is not None and weight_kg2 is not None:
        return 1 if abs(weight_kg1 - weight_kg2) < 1e-2 else 0
    return 0


def compare_dimensions(dim1, dim2):
    dim1_parsed = parse_dimensions(dim1)
    dim2_parsed = parse_dimensions(dim2)
    if not dim1_parsed or not dim2_parsed:
        return 0
    matching_keys = set(dim1_parsed.keys()).intersection(set(dim2_parsed.keys()))
    matching_score = sum(1 for key in matching_keys if abs(dim1_parsed[key] - dim2_parsed[key]) < 1e-2)
    total_keys = len(set(dim1_parsed.keys()).union(set(dim2_parsed.keys())))
    return matching_score / total_keys


def calculate_similarity(details1, details2, title1, title2, desc1, desc2):
    score = 0
    total_keys = len(details1.keys())
    details_comparison = []
    for key in details1.keys():
        if key in details2:
            value1 = str(details1[key])
            value2 = str(details2[key])
            if 'weight' in key.lower():
                match_score = compare_weights(value1, value2)
                details_comparison.append(f"{key}: {value1} vs {value2} -> Match: {match_score}")
                score += match_score
            elif 'dimension' in key.lower() or key.lower() == 'product dimensions':
                match_score = compare_dimensions(value1, value2)
                details_comparison.append(f"{key}: {value1} vs {value2} -> Match Score: {match_score}")
                score += match_score
            else:
                match_score = tokenized_similarity(value1, value2)
                details_comparison.append(f"{key}: {value1} vs {value2} -> Match Score: {match_score}")
                score += match_score
    if total_keys > 0:
        details_score = (score / total_keys) * 100
    else:
        details_score = 0
    title_score, title1_tokens, title2_tokens, title_intersection = title_similarity(title1, title2)
    title_comparison = f"Title Tokens (Target): {title1_tokens}\nTitle Tokens (Competitor): {title2_tokens}\nCommon Tokens: {title_intersection}\nScore: {title_score}"
    desc_score, desc1_tokens, desc2_tokens, desc_intersection = description_similarity(desc1, desc2)
    desc_comparison = f"Description Tokens (Target): {desc1_tokens}\nDescription Tokens (Competitor): {desc2_tokens}\nCommon Tokens: {desc_intersection}\nScore: {desc_score}"

    return details_score, title_score, desc_score, details_comparison, title_comparison, desc_comparison


def calculate_weighted_score(details_score, title_score, desc_score):
    weighted_score = 0.5 * details_score + 0.4 * title_score + 0.1 * desc_score
    return weighted_score


def calculate_cpi_score(price, competitor_prices):
    percentile = 100 * (competitor_prices < price).mean()
    cpi_score = 10 - (percentile / 10)
    return cpi_score

def calculate_cpi_score_updated(target_price, competitor_prices):
    # Compute distances from the target price
    distances = np.abs(competitor_prices - target_price)

    # Define a weighting function: closer prices get higher weights
    max_distance = np.max(distances)
    if max_distance == 0:
        weights = np.ones_like(distances)
    else:
        weights = 1 - (distances / max_distance)

    # Calculate the weighted average of competitor prices
    weighted_average_price = np.average(competitor_prices, weights=weights)

    # Calculate CPI Score
    if weighted_average_price > 0:
        percentile = 100 * (competitor_prices < weighted_average_price).mean()
    else:
        percentile = 100

    cpi_score = 10 - (percentile / 10)
    return cpi_score


def extract_brand_from_title(title):
    if pd.isna(title) or not title:
        return 'unknown'
    return title.split()[0].lower()


def extract_style(title):
    title = str(title)
    style_pattern = r"\b(\d+)\s*(inches?|in|inch|\"|''|'\s*'\s*)\b"
    style_match = re.search(style_pattern, title.lower())

    if style_match:
        number = style_match.group(1)
        return f"{number} Inch"

    style_pattern_with_quote = r"\b(\d+)\s*(''{1,2})"
    style_match = re.search(style_pattern_with_quote, title.lower())

    if style_match:
        number = style_match.group(1)
        return f"{number} Inch"
    return None


def extract_size(title):
    title = str(title)
    size_patterns = {
        'Twin XL': r'\btwin[-\s]xl\b',
        'Queen': r'\bqueen\b',
        'Full': r'\b(full|double)\b',
        'Twin': r'\btwin\b',
        'King': r'\bking\b'
    }

    title_lower = title.lower()

    for size, pattern in size_patterns.items():
        if re.search(pattern, title_lower):
            return size

    return None

show_features_df = None
@st.cache_data
def load_and_preprocess_data():
    global show_features_df

    # Load data only once and perform preprocessing steps
    df_serp = pd.read_csv("C:\\Users\\bande\\Streamlit App\\AMZ_SERPDATA_MATTRESS(Modified).csv", on_bad_lines='skip')
    df_scrapped = pd.read_csv("C:\\Users\\bande\\Streamlit App\\final_scraped_mattress_updated.csv", on_bad_lines='skip')

    df_serp['asin'] = df_serp['asin'].str.upper()
    df_scrapped['ASIN'] = df_scrapped['ASIN'].str.upper()

    # Merge and clean up
    df_serp_cleaned = df_serp.drop_duplicates(subset='asin')
    df_scrapped_cleaned = df_scrapped.drop_duplicates(subset='ASIN')
    df_merged_cleaned = pd.merge(df_scrapped_cleaned, df_serp_cleaned[['asin', 'product_title', 'brand']],
                                 left_on='ASIN', right_on='asin', how='left')
    df_merged_cleaned = df_merged_cleaned.drop('asin', axis=1)

    # Load additional dataset for time-series analysis
    df2 = pd.read_csv("C:\\Users\\bande\\Streamlit App\\combined_asin_price_data.csv")
    # df2 = df2.drop_duplicates(subset=['asin'], keep='first')
    df2['asin'] = df2['asin'].str.upper()
    df_merged_cleaned['ASIN'] = df_merged_cleaned['ASIN'].str.upper()

    # Merge competitor prices with df2
    df = pd.merge(df_merged_cleaned, df2[['asin', 'price', 'date']], left_on='ASIN', right_on='asin', how='left')
    # Parse the 'Product Details', 'Glance Icon Details', 'Option', and 'Drop Down' columns
    df['Product Details'] = df['Product Details'].apply(parse_dict_str)
    df['Glance Icon Details'] = df['Glance Icon Details'].apply(parse_dict_str)

    df['Style'] = df['product_title'].apply(extract_style)
    df['Size'] = df['product_title'].apply(extract_size)

    def update_product_details(row):
        details = row['Product Details']
        details['Style'] = row['Style']
        details['Size'] = row['Size']
        return details

    df['Product Details'] = df.apply(update_product_details, axis=1)

    def extract_dimensions(details):
        # Check if 'Product Dimensions' exists in the dictionary
        if isinstance(details, dict):
            return details.get('Product Dimensions', None)
        return None

    # Create a new column 'Product Dimensions' by extracting from 'Product Details'
    df['Product Dimensions'] = df['Product Details'].apply(extract_dimensions)

    reference_df = pd.read_csv('C:\\Users\\bande\\Streamlit App\\product_dimension_size_style_reference.csv')

    merged_df = df.merge(reference_df, on='Product Dimensions', how='left', suffixes=('', '_ref'))

    # Fill missing values in 'Size' and 'Style' columns with the values from the reference DataFrame
    merged_df['Size'] = merged_df['Size'].fillna(merged_df['Size_ref'])
    merged_df['Style'] = merged_df['Style'].fillna(merged_df['Style_ref'])


    merged_df['date'] = pd.to_datetime(merged_df['date'])
    df_ext = merged_df[merged_df['date']==merged_df['date'].max()]

    df_ext['size_extracted'] = df_ext['Size'].notnull()
    df_ext['style_extracted'] = df_ext['Style'].notnull()

    df_ext['extraction_scenario'] = df_ext.apply(
        lambda row: (
            'Both Size and Style Extracted' if row['size_extracted'] and row['style_extracted'] else
            'Neither Size nor Style Extracted' if not row['size_extracted'] and not row['style_extracted'] else
            'Only Size Extracted' if row['size_extracted'] and not row['style_extracted'] else
            'Only Style Extracted'
        ), axis=1
    )

    scenario_counts = df_ext['extraction_scenario'].value_counts()

    total_products = df_ext.shape[0]
    scenario_percentages = (scenario_counts / total_products) * 100

    for scenario, count in scenario_counts.items():
        percentage = scenario_percentages[scenario]
        print(f"{scenario}: {count} products ({percentage:.2f}%)")

    output_columns = ['ASIN', 'product_title', 'Drop Down', 'Product Details', 'Glance Icon Details', 'Description',
                      'Option', 'Rating', 'Review Count','Size','Style','Product Dimensions','Size_ref','Style_ref']

    both_extracted_df = df_ext[df_ext['extraction_scenario'] == 'Both Size and Style Extracted']
    only_size_extracted_df = df_ext[df_ext['extraction_scenario'] == 'Only Size Extracted']
    only_style_extracted_df = df_ext[df_ext['extraction_scenario'] == 'Only Style Extracted']
    neither_extracted_df = df_ext[df_ext['extraction_scenario'] == 'Neither Size nor Style Extracted']

    both_extracted_df.to_csv('Size_and_Style_Extracted.csv', columns=output_columns, index=False)
    only_size_extracted_df.to_csv('Size_Extracted.csv', columns=output_columns, index=False)
    only_style_extracted_df.to_csv('Style_Extracted.csv', columns=output_columns, index=False)
    neither_extracted_df.to_csv('Size_Nor_Style_extracted.csv', columns=output_columns, index=False)

    df = merged_df.copy()

    df_d = df[['ASIN', 'Size', 'Style']].drop_duplicates(subset=['ASIN'])
    combinations_df = df_d[['Size', 'Style']]
    combination_counts = combinations_df.value_counts()
    combination_counts_df = combination_counts.reset_index(name='count')
    combination_counts_df = combination_counts_df.sort_values(by='count', ascending=False)
    combination_counts_df.to_csv('combination_zero.csv')



    df.to_csv('processed_data.csv',index=False)
    show_features_df = df.copy()
    return df

def check_compulsory_features_match(target_details, compare_details, compulsory_features):

    for feature in compulsory_features:
        if feature not in target_details:
            return False
        if feature not in compare_details:
            return False
        target_value = str(target_details[feature]).lower()
        compare_value = str(compare_details[feature]).lower()
        if target_value != compare_value:
            return False

    return True

def find_similar_products(asin, price_min, price_max, df, compulsory_features, same_brand_option):
    print(compulsory_features)
    df['identified_brand'] = df['product_title'].apply(extract_brand_from_title)

    target_product = df[df['ASIN'] == asin].iloc[0]
    print(target_product)
    print("product")
    print(type(target_product))
    # print(type(target_product('Option', {})))
    target_details = {**target_product['Product Details'], **target_product['Glance Icon Details']}
    print(target_details)

    target_brand = target_product['identified_brand']
    target_thickness = extract_thickness(target_details.get('Product Dimensions', ''))
    target_title = str(target_product['product_title']).lower()
    target_desc = str(target_product['Description']).lower()

    similarities = []
    unique_asins = set()
    seen_combinations = set()

    for index, row in df.iterrows():
        if row['ASIN'] == asin:
            continue
        compare_brand = row['identified_brand']
        if same_brand_option == 'only' and compare_brand != target_brand:
            continue
        if same_brand_option == 'omit' and compare_brand == target_brand:
            continue
        if price_min <= row['price'] <= price_max:
            # print(type(row('Option',{})))
            compare_details = {**row['Product Details'], **row['Glance Icon Details']}

            compare_thickness = extract_thickness(compare_details.get('Product Dimensions', ''))
            compare_title = str(row['product_title']).lower()
            compare_desc = str(row['Description']).lower()


            compulsory_match = check_compulsory_features_match(target_details, compare_details, compulsory_features)

            if compulsory_match:
                asin = row['ASIN']
                # Adjust the combination to include title, price, and product details
                combination = (compare_title, row['price'], str(compare_details))

                if combination not in seen_combinations:  # Check if this combination has been seen before
                    if asin not in unique_asins:
                        details_score, title_score, desc_score, details_comparison, title_comparison, desc_comparison = calculate_similarity(
                            target_details, compare_details, target_title, compare_title, target_desc, compare_desc
                        )
                        weighted_score = calculate_weighted_score(details_score, title_score, desc_score)
                        if weighted_score >= 0:
                            similarities.append(
                                (asin, row['product_title'], row['price'], weighted_score, details_score,
                                 title_score, desc_score, compare_details, details_comparison, title_comparison,
                                 desc_comparison, compare_brand))
                        unique_asins.add(asin)
                        seen_combinations.add(combination)  # Mark this combination as seen

    similarities = sorted(similarities, key=lambda x: x[3], reverse=True)
    print(len(similarities))
    similarities_df = pd.DataFrame(similarities, columns=[
        'ASIN', 'Product Title', 'Price', 'Weighted Score', 'Details Score',
        'Title Score', 'Description Score', 'Compare Details', 'Details Comparison',
        'Title Comparison', 'Description Comparison', 'Brand'
    ])
    similarities_df.to_csv('similarity_df.csv')
    return similarities

def run_analysis(asin, price_min, price_max, target_price, compulsory_features, same_brand_option, df):
    similar_products = find_similar_products(asin, price_min, price_max, df, compulsory_features, same_brand_option)
    prices = [p[2] for p in similar_products]
    competitor_prices = np.array(prices)
    cpi_score = calculate_cpi_score(target_price, competitor_prices)
    cpi_score_dynamic = calculate_cpi_score_updated(target_price, competitor_prices)
    target_product = df[df['ASIN'] == asin].iloc[0]
    num_competitors_found = len(similar_products)
    target_product = df[df['ASIN'] == asin].iloc[0]
    size = target_product['Product Details'].get('Size', 'N/A')
    product_dimension = target_product['Product Details'].get('Product Dimensions', 'N/A')

    # Create a DataFrame to store competitor details
    competitor_details_df = pd.DataFrame(similar_products, columns=[
        'ASIN', 'Title', 'Price', 'Weighted Score', 'Details Score',
        'Title Score', 'Description Score', 'Product Details',
        'Details Comparison', 'Title Comparison', 'Description Comparison', 'Brand'
    ])

    # Filter the dataframe to include only the required columns
    competitor_details_df = competitor_details_df[['ASIN', 'Title', 'Price', 'Product Details', 'Brand']]

    # Add matching compulsory features
    competitor_details_df['Matching Features'] = competitor_details_df['Product Details'].apply(
        lambda details: {feature: details.get(feature, 'N/A') for feature in compulsory_features}
    )

    return asin, target_price, cpi_score, num_competitors_found, size, product_dimension, prices, competitor_details_df, cpi_score_dynamic


def show_features():

    if show_features_df is None:
        st.error("Error", "DataFrame is not initialized.")
        return

    if asin not in show_features_df['ASIN'].values:
        st.error("Error", "ASIN not found.")
        return
    target_product = show_features_df[show_features_df['ASIN'] == asin].iloc[0]
    product_details = target_product['Product Details']  # **target_product['Glance Icon Details']}

    st.subheader(f"Product Details for ASIN: {asin}")

    # Display product details
    st.text("Product Details:")
    st.text(format_details(product_details))

    st.subheader("Select Compulsory Features:")
    
    # Display checkboxes for each product detail feature
    compulsory_features_vars = {}
    for feature in product_details.keys():
        var = st.checkbox(f"Include {feature}")
        compulsory_features_vars[feature] = var

    return compulsory_features_vars

def perform_scatter_plot(asin, target_price, price_min, price_max, compulsory_features, same_brand_option, df):
    # Find similar products
    similar_products = find_similar_products(asin, price_min, price_max, df, compulsory_features, same_brand_option)

    # Retrieve target product information
    target_product = df[df['ASIN'] == asin].iloc[0]
    target_title = str(target_product['product_title']).lower()
    target_desc = str(target_product['Description']).lower()
    target_details = target_product['Product Details']

    # Calculate similarity scores for the target product
    details_score, title_score, desc_score, details_comparison, title_comparison, desc_comparison = calculate_similarity(
        target_details, target_details, target_title, target_title, target_desc, target_desc
    )
    weighted_score = calculate_weighted_score(details_score, title_score, desc_score)
    
    target_product_entry = (
        asin, target_product['product_title'], target_price, weighted_score, details_score,
        title_score, desc_score, target_details, details_comparison, title_comparison, desc_comparison
    )

    # Ensure the target product is not included in the similar products list
    similar_products = [prod for prod in similar_products if prod[0] != asin]
    similar_products.insert(0, target_product_entry)

    # Extract price and weighted scores from similar products
    prices = [p[2] for p in similar_products]
    weighted_scores = [p[3] for p in similar_products]
    indices = range(len(similar_products))

    # Create scatter plot using Matplotlib
    fig, ax1 = plt.subplots(figsize=(12, 8))
    scatter = ax1.scatter(indices, prices, c=weighted_scores, cmap='viridis', s=50, label='Similar Products')
    
    # Plot the target product separately
    ax1.scatter([0], [target_price], c='red', marker='*', s=200, label='Target Product')

    ax1.set_xlabel('Index')
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f"Comparison of Similar Products to ASIN: {asin}")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Weighted Scores')
    ax2.set_ylim(0, 100)
    ax2.set_yticks(np.linspace(0, 100, 6))
    ax2.set_xlim(ax1.get_xlim())
    ax1.legend(loc='upper right')

    # Add a color bar for weighted scores
    plt.colorbar(scatter, ax=ax1, label='Weighted Score')

    # Display the scatter plot in Streamlit
    st.pyplot(fig)

    # Equivalent of the side panel in Streamlit to display detailed information
    st.subheader("Weighted Score Details:")
    
    # Allow the user to select a product to display detailed information
    selected_index = st.selectbox("Select a product index to view details:", options=list(range(len(similar_products))))

    # Retrieve the selected product details
    selected_product = similar_products[selected_index]
    target_asin = selected_product[0]

    # Find the target product's size and style
    target_product = df[df['ASIN'] == target_asin].iloc[0]
    target_size = target_product['Size']
    target_style = target_product['Style']

    # Analyze size/style combinations for competitors
    df_d = df[['ASIN', 'Size', 'Style']].drop_duplicates(subset=['ASIN'])
    combinations_df = df_d[['Size', 'Style']]
    combination_counts = combinations_df.value_counts()
    combination_counts_df = combination_counts.reset_index(name='count')
    combination_counts_df = combination_counts_df.sort_values(by='count', ascending=False)

    # Filter for competitors with the same size and style
    filtered_df = df[(df['Size'] == target_size) & (df['Style'] == target_style)]
    filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')
    filtered_df = filtered_df[filtered_df['date'] == filtered_df['date'].max()]
    price_null_count = filtered_df['price'].isnull().sum()

    # Display total combinations
    total_combinations = len(combination_counts_df)

    # Display the count of each combination
    combination_details = "\n".join([
        f"Size: {row['Size']}, Style: {row['Style']} - Count: {row['count']}"
        for _, row in combination_counts_df.iterrows()
    ])

    target_combination_count = combination_counts_df[
        (combination_counts_df['Size'] == target_size) & (combination_counts_df['Style'] == target_style)
    ]['count'].values[0]

    # Display detailed information in Streamlit
    st.write(f"Competitor Count: {len(similar_products)}")
    st.write(f"Target Product's Size: {target_size}")
    st.write(f"Target Product's Style: {target_style}")
    st.write(f"Count of Target Size-Style Combination On The Day Of Analysis: {target_combination_count}")
    st.write(f"Number of Competitors With Null Price: {price_null_count}")

    # CPI Score Polar Plot
    competitor_prices = np.array(prices)
    cpi_score = calculate_cpi_score(target_price, competitor_prices)
    dynamic_cpi_score = calculate_cpi_score_updated(target_price, competitor_prices)

    st.subheader("CPI Score Comparison")

    # Create CPI radar charts (one for static and one for dynamic CPI)
    fig_cpi, (ax_cpi, ax_dynamic_cpi) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'polar': True})

    categories = [''] * 10
    angles = np.linspace(0, np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    values = [0] * 10
    values += values[:1]

    # Plot original CPI score
    ax_cpi.fill(angles, values, color='grey', alpha=0.25)
    score_angle = (cpi_score / 10) * np.pi
    ax_cpi.plot([0, score_angle], [0, 10], color='blue', linewidth=2, linestyle='solid')
    ax_cpi.set_title("CPI Score")

    # Plot dynamic CPI score
    ax_dynamic_cpi.fill(angles, values, color='grey', alpha=0.25)
    dynamic_score_angle = (dynamic_cpi_score / 10) * np.pi
    ax_dynamic_cpi.plot([0, dynamic_score_angle], [0, 10], color='green', linewidth=2, linestyle='solid')
    ax_dynamic_cpi.set_title("Dynamic CPI Score")

    # Display CPI score plots
    st.pyplot(fig_cpi)

def process_date(df2, asin, date_str, price_min, price_max, compulsory_features, same_brand_option):
    """
    This function processes data for a single date and returns the results.
    """
    df_combined = df2.copy()
    df_combined['date'] = pd.to_datetime(df_combined['date'], format='%Y-%m-%d')
    df_current_day = df_combined[df_combined['date'] == date_str]

    if df_current_day.empty:
        st.error(f"No data found for date: {date_str}")
        return None

    try:
        target_price = df_current_day[df_current_day['asin'] == asin]['price'].values[0]
    except IndexError:
        st.error(f"ASIN {asin} not found for date {date_str}")
        return None

    # Calling run_analysis (assuming it's available and properly defined)
    result = run_analysis(asin, price_min, price_max, target_price, compulsory_features, same_brand_option, df_current_day)

    # Calculate the number of products with missing or invalid prices
    daily_null_count = df_current_day['price'].isna().sum() + (df_current_day['price'] == 0).sum() + (df_current_day['price'] == '').sum()

    return {
        'date': date_str,
        'result': result,
        'daily_null_count': daily_null_count,
        'num_competitors_found': result[3]
    }

def calculate_and_plot_cpi(df2, asin_list, start_date, end_date, price_min, price_max, compulsory_features, same_brand_option):
    """
    Perform time-series analysis for ASINs between the given start and end dates.
    """
    asin = asin_list[0]
    start_date = datetime.strptime(str(start_date), '%Y-%m-%d')
    end_date = datetime.strptime(str(end_date), '%Y-%m-%d')

    all_results = []
    competitor_count_per_day = []
    null_price_count_per_day = []

    current_date = start_date
    dates_to_process = []

    # Prepare list of dates to process
    while current_date <= end_date:
        dates_to_process.append(current_date)
        current_date += timedelta(days=1)

    # Use ThreadPoolExecutor for parallel processing if possible (though Streamlit typically runs sequentially)
    for current_date in dates_to_process:
        result = process_date(df2, asin, pd.to_datetime(current_date), price_min, price_max, compulsory_features, same_brand_option)
        if result is not None:
            daily_results = result['result'][:-1]
            daily_null_count = result['daily_null_count']
            num_competitors_found = result['num_competitors_found']

            all_results.append((result['date'], *daily_results))
            competitor_count_per_day.append(num_competitors_found)
            null_price_count_per_day.append(daily_null_count)

    # Create result DataFrame
    result_df = pd.DataFrame(all_results,
                             columns=['Date', 'ASIN', 'Target Price', 'CPI Score', 'Number Of Competitors Found',
                                      'Size', 'Product Dimension', 'Competitor Prices', 'Dynamic CPI'])

    # Display the result dataframe in Streamlit
    st.subheader("Analysis Results")
    st.dataframe(result_df)

    # Load additional data for merging (ads data, for example)
    try:
        napqueen_df = pd.read_csv("C:\\Users\\bande\\Streamlit App\\ads_data_sep15.csv")
        napqueen_df['date'] = pd.to_datetime(napqueen_df['date'], format='%d-%m-%Y', errors='coerce')
        napqueen_df = napqueen_df.rename(columns={'date': 'Date', 'asin': 'ASIN'})

        result_df['Date'] = pd.to_datetime(result_df['Date'], format='%Y-%m-%d')
        result_df = pd.merge(result_df, napqueen_df[['Date', 'ASIN', 'ad_spend', 'orderedunits']], on=['Date', 'ASIN'], how='left')

        st.success("Merging successful! Displaying the merged dataframe:")
        st.dataframe(result_df)

    except KeyError as e:
        st.error(f"KeyError: {e} - Likely missing columns during merging.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Plot the results using Streamlit's built-in or Matplotlib
    st.subheader("Time-Series Analysis Results")

    if not result_df.empty:
        st.write("CPI Score Over Time:")
        st.line_chart(result_df.set_index('Date')['CPI Score'])

        st.write("Target Price Over Time:")
        st.line_chart(result_df.set_index('Date')['Target Price'])

        st.write("Ad Spend Over Time:")
        st.line_chart(result_df.set_index('Date')['ad_spend'])

        st.write("Ordered Units Over Time:")
        st.line_chart(result_df.set_index('Date')['orderedunits'])

    else:
        st.warning("No data available for plotting.")


def plot_competitor_vs_null_analysis(competitor_count_per_day, null_price_count_per_day, start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Competitors Found', color='tab:blue')
    ax1.plot(dates, competitor_count_per_day, color='tab:blue', label='Competitors Found')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Null Price Values', color='tab:orange')
    ax2.plot(dates, null_price_count_per_day, color='tab:orange', label='Null Price Values')

    # Set date format and tick locator for x-axis
    ax1.xaxis.set_major_locator(mdates.DayLocator())  # Set tick locator to daily
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))  # Format date labels
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    plt.title('Competitors Found vs Null Price Values Over Time')
    fig.tight_layout()
    st.pyplot(fig)


def plot_results(result_df, asin_list, start_date, end_date):


    result_df.to_csv('plot_csv.csv')


    for asin in asin_list:
        asin_results = result_df[result_df['ASIN'] == asin]
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.set_xlabel('Date')
        ax1.set_ylabel('CPI Score', color='tab:blue')
        ax1.plot(pd.to_datetime(asin_results['Date']), asin_results['CPI Score'], label='CPI Score', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        ax1.xaxis.set_major_locator(mdates.DayLocator())
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_xlim(start_date, end_date)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Price', color='tab:orange')
        ax2.plot(pd.to_datetime(asin_results['Date']), asin_results['Target Price'], label='Price', linestyle='--',
                 color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.set_ylabel('Ad Spend', color='tab:green')
        ax3.plot(pd.to_datetime(asin_results['Date']), asin_results['ad_spend'], label='ad_spend', linestyle='-.',
                 color='tab:green')
        ax3.tick_params(axis='y', labelcolor='tab:green')

        ax4 = ax1.twinx()
        ax4.spines['right'].set_position(('outward', 120))
        ax4.set_ylabel('Ordered Units', color='tab:purple')
        ax4.plot(pd.to_datetime(asin_results['Date']), asin_results['orderedunits'], label='Ordered Units',
                 color='tab:purple')
        ax4.tick_params(axis='y', labelcolor='tab:purple')

        # ax5 = ax1.twinx()
        # ax5.spines['right'].set_position(('outward', 180))
        # ax5.set_ylabel('Dynamic CPI', color='tab:yellow')
        # ax5.plot(pd.to_datetime(asin_results['Date']), asin_results['Dynamic CPI'], label='Dynamic CPI',
        #          color='tab:yellow')
        # ax5.tick_params(axis='y', labelcolor='tab:yellow')

        plt.title(f'CPI Score, Price, Ad Spend and Ordered Units Over Time for ASIN {asin}')
        fig.tight_layout()
        st.pyplot(fig)


def get_distribution_date(result_df, asin):
     # Streamlit's date input widget
    selected_date = st.date_input("Select Distribution Date", datetime.now())

    if st.button("Plot Distribution Graph"):
        plot_distribution_graph(result_df, asin, selected_date)

def plot_distribution_graph(result_df, asin, selected_date):
    asin_results = result_df[result_df['ASIN'] == asin]
    selected_data = asin_results[asin_results['Date'] == selected_date]

    if selected_data.empty:
        st.error("Error", "No data available for the selected date.")
        return

    target_price = selected_data['Target Price'].values[0]
    competitor_prices = selected_data['Competitor Prices'].values[0]

    competitor_prices = [float(price) for price in competitor_prices]

    fig = make_subplots(rows=1, cols=1)

    # Competitor Prices Histogram
    fig.add_trace(
        go.Histogram(x=competitor_prices, name='Competitors', marker_color='blue', opacity=0.7, hoverinfo='x+y',
                     xbins=dict(size=10)), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=[target_price], y=[0], mode='markers', marker=dict(color='purple', size=12), name='Target Price',
                   hoverinfo='x'), row=1, col=1)

    fig.update_layout(barmode='overlay', title_text=f'Price Distribution on {selected_date.date()}', showlegend=True)
    fig.update_traces(marker_line_width=1.2, marker_line_color='black')

    fig.update_xaxes(title_text='Price')
    fig.update_yaxes(title_text='Frequency')

    st.plotly_chart(fig)

def run_analysis_button(df, asin, price_min, price_max, target_price, start_date, end_date, same_brand_option, compulsory_features):
    st.write("Inside Analysis")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df_recent = df[df['date'] == df['date'].max()]
    df_recent = df_recent.drop_duplicates(subset=['asin'])

    # Ensure that ASIN exists in the dataset
    if asin not in df['ASIN'].values:
        st.error("Error: ASIN not found.")
        return

    # Extract the product information for the target ASIN
    target_product = df[df['ASIN'] == asin].iloc[0]
    target_brand = target_product['brand'].lower() if 'brand' in target_product else None

    if target_brand is None:
        try:
            target_brand = target_product['Product Details'].get('Brand', '').lower()
        except:
            pass

    st.write(f"Brand: {target_brand}")

    # Check if we should perform time-series analysis (only if brand == 'napqueen' and dates are provided)
    if target_brand and target_brand.lower() == "napqueen" and start_date and end_date:
        perform_scatter_plot(asin, target_price, price_min, price_max, compulsory_features, same_brand_option, df_recent)
        calculate_and_plot_cpi(df, [asin], start_date, end_date, price_min, price_max, compulsory_features, same_brand_option)
    else:
        # Perform scatter plot only
        perform_scatter_plot(asin, target_price, price_min, price_max, compulsory_features, same_brand_option, df_recent)


# Load data globally before starting the Streamlit app
df = load_and_preprocess_data()

# Streamlit UI for ASIN Competitor Analysis
st.title("ASIN Competitor Analysis")

# Input fields for ASIN, price range, and target price
asin = st.text_input("Enter ASIN").upper()
price_min = st.number_input("Price Min", value=0.0)
price_max = st.number_input("Price Max", value=1000.0)
target_price = st.number_input("Target Price", value=0.0)

# Date input for time-series analysis (optional)
start_date = st.date_input("Start Date", value=pd.to_datetime('2023-01-01'))
end_date = st.date_input("End Date", value=pd.to_datetime('2023-12-31'))

# Radio buttons for same brand option
same_brand_option = st.radio("Same Brand Option", ('all', 'only', 'omit'))

# Button to show features based on entered ASIN
if st.button("Show Features"):
    if asin in df['ASIN'].values:
        show_features(asin)  # Show features for the entered ASIN
    else:
        st.error("ASIN not found in dataset.")

# Automatically display checkboxes for each product detail feature (if ASIN exists)
compulsory_features_vars = {}
if asin in df['ASIN'].values:
    product_details = df[df['ASIN'] == asin].iloc[0]['Product Details']
    st.write("Select compulsory features:")
    for feature in product_details.keys():
        compulsory_features_vars[feature] = st.checkbox(f"Include {feature}")

# Collect selected compulsory features
compulsory_features = [feature for feature, selected in compulsory_features_vars.items() if selected]

# Button to run analysis - this triggers the run_analysis_button function
if st.button("Analyze"):
    # Call the run_analysis_button function with all the inputs from Streamlit
    run_analysis_button(df, asin, price_min, price_max, target_price, start_date, end_date, same_brand_option, compulsory_features)