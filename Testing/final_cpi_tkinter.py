import tkinter as tk
from tkinter import ttk, messagebox
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
import os


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
def load_and_preprocess_data():
    global show_features_df

    # Load data only once and perform preprocessing steps
    df_serp = pd.read_csv("AMZ_SERPDATA_MATTRESS(Modified).csv", on_bad_lines='skip')
    df_scrapped = pd.read_csv("final_scraped_mattress_updated.csv", on_bad_lines='skip')

    df_serp['asin'] = df_serp['asin'].str.upper()
    df_scrapped['ASIN'] = df_scrapped['ASIN'].str.upper()

    # Merge and clean up
    df_serp_cleaned = df_serp.drop_duplicates(subset='asin')
    df_scrapped_cleaned = df_scrapped.drop_duplicates(subset='ASIN')
    df_merged_cleaned = pd.merge(df_scrapped_cleaned, df_serp_cleaned[['asin', 'product_title', 'brand']],
                                 left_on='ASIN', right_on='asin', how='left')
    df_merged_cleaned = df_merged_cleaned.drop('asin', axis=1)

    # Load additional dataset for time-series analysis
    df2 = pd.read_csv("combined_asin_price_data.csv")
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

    reference_df = pd.read_csv('product_dimension_size_style_reference.csv')

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
    #
    #
    # df_d = df[['ASIN', 'Size', 'Style']].drop_duplicates()
    # combinations_df = df_d[['Size', 'Style']]
    # combination_counts = combinations_df.value_counts()
    # combination_counts_df = combination_counts.reset_index(name='count')
    # combination_counts_df = combination_counts_df.sort_values(by='count', ascending=False)
    # my_size = df[df['ASIN'] == asin]['Size'].values[0]
    # my_style = df[df['ASIN'] == asin]['Style'].values[0]
    #
    # filtered_df = df[(df['Size'] == my_size) & (df['Style'] == my_style)]
    # filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')
    # filtered_df = filtered_df[filtered_df['date'] == filtered_df['date'].max()]
    # filtered_df.to_csv('test.csv', index=False)
    # # Getting the count of the specific size and style combination
    # my_combination_count = combination_counts_df[(combination_counts_df['Size'] == my_size) &(combination_counts_df['Style'] == my_style)
    # ]
    # print(my_combination_count)
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

    # Extract Product Dimension and Matching Features
    competitor_details_df['Product Dimension'] = competitor_details_df['Product Details'].apply(
        lambda details: details.get('Product Dimensions', 'N/A'))
    
    # Add matching compulsory features
    competitor_details_df['Matching Features'] = competitor_details_df['Product Details'].apply(
        lambda details: {feature: details.get(feature, 'N/A') for feature in compulsory_features}
    )

    # Filter the dataframe to include only the required columns
    competitor_details_df = competitor_details_df[['ASIN', 'Title', 'Price', 'Product Dimension', 'Brand', 'Matching Features']]

    return asin, target_price, cpi_score, num_competitors_found, size, product_dimension, prices, competitor_details_df, cpi_score_dynamic


def show_features():
    asin = asin_entry.get()
    print(asin)

    if show_features_df is None:
        messagebox.showerror("Error", "DataFrame is not initialized.")
        return

    if asin not in show_features_df['ASIN'].values:
        messagebox.showerror("Error", "ASIN not found.")
        return
    target_product = show_features_df[show_features_df['ASIN'] == asin].iloc[0]
    product_details = target_product['Product Details']  # **target_product['Glance Icon Details']}

    print("Product Details Keys:", product_details.keys())

    for widget in feature_frame.winfo_children():
        widget.destroy()

    tk.Label(feature_frame, text="Select Compulsory Features:", font=("Helvetica", 10, "bold")).pack(anchor='w')
    for feature in product_details.keys():
        # if feature.lower() == 'color':
        # continue  # Skip the 'Color' key
        var = tk.BooleanVar()
        print(f"Adding checkbox for feature: {feature}")  # Debug statement
        tk.Checkbutton(feature_frame, text=feature, variable=var).pack(anchor='w')
        compulsory_features_vars[feature] = var
    for widget in product_details_frame.winfo_children():
        widget.destroy()
    tk.Label(product_details_frame, text="Product Details:", font=("Helvetica", 10, "bold")).pack(anchor='w')
    details_text = tk.Text(product_details_frame, wrap=tk.WORD, height=10)
    details_text.pack(fill=tk.BOTH, expand=True)
    details_text.insert(tk.END, format_details(product_details))
    details_text.config(state=tk.DISABLED)

def perform_scatter_plot(asin, target_price, price_min, price_max, compulsory_features, same_brand_option,df):
    similar_products = find_similar_products(asin, price_min, price_max, df, compulsory_features, same_brand_option)

    target_product = df[df['ASIN'] == asin].iloc[0]
    target_title = str(target_product['product_title']).lower()
    target_desc = str(target_product['Description']).lower()
    target_details = target_product['Product Details']
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

    # Create a new Toplevel window for the scatter plot
    scatter_window = tk.Toplevel(root)
    scatter_window.title("Scatter Plot Analysis")

    result_frame = ttk.Frame(scatter_window, padding=(10, 10))
    result_frame.pack(fill=tk.BOTH, expand=True)
    result_frame.columnconfigure(0, weight=1)
    result_frame.rowconfigure(0, weight=1)
    result_frame.rowconfigure(1, weight=1)
    result_frame.rowconfigure(2, weight=1)

    plot_frame = ttk.LabelFrame(result_frame, text="Scatter Plot", padding=(10, 10))
    plot_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

    side_panel_frame = ttk.Frame(result_frame, padding=(10, 10))
    side_panel_frame.grid(row=0, column=1, sticky='nsw', padx=10, pady=10)

    side_panel_label = ttk.Label(side_panel_frame, text="Weighted Score Details:", font=("Helvetica", 10, "bold"))
    side_panel_label.pack(anchor='nw')

    side_panel_text = tk.Text(side_panel_frame, wrap=tk.WORD, height=20, width=40)
    side_panel_text.pack(fill=tk.BOTH, expand=True)
    side_panel_text.config(state=tk.DISABLED)

    fig, ax1 = plt.subplots(figsize=(12, 8))
    prices = [p[2] for p in similar_products]
    weighted_scores = [p[3] for p in similar_products]
    indices = range(len(similar_products))

    scatter = ax1.scatter(indices, prices, c=weighted_scores, cmap='viridis', s=50, label='Similar Products')
    target_scatter = ax1.scatter([0], [target_price], c='red', marker='*', s=200, label='Target Product')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Price$')
    ax1.set_title(f"Comparison of Similar Products to ASIN: {asin}")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Weighted Scores')
    ax2.set_ylim(0, 100)
    ax2.set_yticks(np.linspace(0, 100, 6))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    ax2.set_xlim(ax1.get_xlim())
    ax1.legend(loc='upper right')

    plt.tight_layout(pad=3.0)
    ax1.grid(True)
    ax2.grid(False)
    plt.colorbar(scatter, ax=ax1, label='Weighted Score', pad=0.08)

    def update_side_panel(sel):
        # Retrieve product details from the selected index in the scatter plot

        product = similar_products[sel.index]
        target_asin = product[0]

        # Find the target product's size and style
        target_product = df[df['ASIN'] == target_asin].iloc[0]
        target_size = target_product['Size']
        target_style = target_product['Style']

        df_d = df[['ASIN', 'Size', 'Style']].drop_duplicates(subset=['ASIN'])
        combinations_df = df_d[['Size', 'Style']]
        combination_counts = combinations_df.value_counts()
        combination_counts_df = combination_counts.reset_index(name='count')
        combination_counts_df = combination_counts_df.sort_values(by='count', ascending=False)
        combination_counts_df.to_csv('combination_one.csv')

        show_features_df

        df_d = show_features_df[['ASIN', 'Size', 'Style']].drop_duplicates(subset=['ASIN'])
        combinations_df = df_d[['Size', 'Style']]
        combination_counts = combinations_df.value_counts()
        combination_counts_df_overall = combination_counts.reset_index(name='count')
        combination_counts_df_overall = combination_counts_df_overall.sort_values(by='count', ascending=False)
        combination_counts_df_overall.to_csv('combination_one_overall.csv')

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

        target_combination_count_overall = combination_counts_df_overall[
            (combination_counts_df_overall['Size'] == target_size) & (combination_counts_df_overall['Style'] == target_style)
            ]['count'].values[0]


        # Format the content to be displayed in the side panel
        side_panel_content = (
            f"Competitor Count: {len(similar_products)}\n"
            f"Target Product's Size: {target_size}\n"
            f"Target Product's Style: {target_style}\n"
            f"Count of Target Size-Style Combination On The Day Of Analysis: {target_combination_count}\n"
            f"Count of Target Size-Style Combination On The One Month File Used For Analysis: {target_combination_count_overall}\n"
            f"Number of Competitors With Null Price: {price_null_count}\n"
        )

        # Display the content in the side panel
        side_panel_text.config(state=tk.NORMAL)
        side_panel_text.delete(1.0, tk.END)
        side_panel_text.insert(tk.END, side_panel_content)
        side_panel_text.config(state=tk.DISABLED)

        # Update the annotation in the scatter plot
        sel.annotation.set_text(
            f"ASIN: {product[0]}\nTitle: {product[1]}\nPrice: ${product[2]:.2f}\n"
            f"Size: {target_size}\nStyle: {target_style}\n"
        )
        sel.annotation.get_bbox_patch().set(fc="yellow", alpha=0.5)

    cursor = mplcursors.cursor(scatter, hover=True)
    cursor.connect("add", lambda sel: update_side_panel(sel))

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # CPI Score Polar Plot
    competitor_prices = np.array(prices)
    cpi_score = calculate_cpi_score(target_price, competitor_prices)
    dynamic_cpi_score = calculate_cpi_score_updated(target_price, competitor_prices)

    # Create a frame for CPI Score comparison
    cpi_frame = ttk.LabelFrame(result_frame, text="CPI Score Comparison", padding=(10, 10))
    cpi_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)

    # Create a figure with two subplots side by side
    fig_cpi, (ax_cpi, ax_dynamic_cpi) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'polar': True})

    # Categories for the radar chart
    categories = [''] * 10
    angles = np.linspace(0, np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    values = [0] * 10
    values += values[:1]

    # Plot the original CPI score on the first subplot (ax_cpi)
    ax_cpi.fill(angles, values, color='grey', alpha=0.25)
    ax_cpi.fill(angles, values, color='grey', alpha=0.5)
    score_angle = (cpi_score / 10) * np.pi
    ax_cpi.plot([0, score_angle], [0, 10], color='blue', linewidth=2, linestyle='solid')
    ax_cpi.fill([0, score_angle, score_angle, 0], [0, 10, 0, 0], color='blue', alpha=0.5)
    ax_cpi.set_ylim(0, 10)
    ax_cpi.set_yticklabels([])
    ax_cpi.set_xticklabels([])
    ax_cpi.text(0, 0, f'{cpi_score:.2f}', horizontalalignment='center', verticalalignment='center', fontsize=20,
                fontweight='bold')
    ax_cpi.set_title("CPI Score")

    # Plot the dynamic CPI score on the second subplot (ax_dynamic_cpi)
    ax_dynamic_cpi.fill(angles, values, color='grey', alpha=0.25)
    ax_dynamic_cpi.fill(angles, values, color='grey', alpha=0.5)
    dynamic_score_angle = (dynamic_cpi_score / 10) * np.pi
    ax_dynamic_cpi.plot([0, dynamic_score_angle], [0, 10], color='green', linewidth=2, linestyle='solid')
    ax_dynamic_cpi.fill([0, dynamic_score_angle, dynamic_score_angle, 0], [0, 10, 0, 0], color='green', alpha=0.5)
    ax_dynamic_cpi.set_ylim(0, 10)
    ax_dynamic_cpi.set_yticklabels([])
    ax_dynamic_cpi.set_xticklabels([])
    ax_dynamic_cpi.text(0, 0, f'{dynamic_cpi_score:.2f}', horizontalalignment='center', verticalalignment='center',
                        fontsize=20,
                        fontweight='bold')
    ax_dynamic_cpi.set_title("Dynamic CPI Score")

    # Adjust layout to avoid clipping of titles
    plt.subplots_adjust(wspace=0.4, top=0.85)

    # Embed the figure in the Tkinter window
    canvas_cpi = FigureCanvasTkAgg(fig_cpi, master=cpi_frame)
    canvas_cpi.draw()
    canvas_cpi.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# def run_analysis(asin, price_min, price_max, target_price, compulsory_features, same_brand_option, df):
#     similar_products = find_similar_products(asin, price_min, price_max, df, compulsory_features, same_brand_option)
#     prices = [p[2] for p in similar_products]
#     competitor_prices = np.array(prices)
#     cpi_score = calculate_cpi_score(target_price, competitor_prices)
#     target_product = df[df['ASIN'] == asin].iloc[0]
#     num_competitors_found = len(similar_products)
#     target_product = df[df['ASIN'] == asin].iloc[0]
#     size = target_product['Product Details'].get('Size', 'N/A')
#     product_dimension = target_product['Product Details'].get('Product Dimensions', 'N/A')
#
#     # Create a DataFrame to store competitor details
#     competitor_details_df = pd.DataFrame(similar_products, columns=[
#         'ASIN', 'Title', 'Price', 'Weighted Score', 'Details Score',
#         'Title Score', 'Description Score', 'Product Details',
#         'Details Comparison', 'Title Comparison', 'Description Comparison', 'Brand'
#     ])
#
#     # Filter the dataframe to include only the required columns
#     competitor_details_df = competitor_details_df[['ASIN', 'Title', 'Price', 'Product Details', 'Brand']]
#
#     # Add matching compulsory features
#     competitor_details_df['Matching Features'] = competitor_details_df['Product Details'].apply(
#         lambda details: {feature: details.get(feature, 'N/A') for feature in compulsory_features}
#     )
#
#     return asin, target_price, cpi_score, num_competitors_found, size, product_dimension, prices, competitor_details_df

# def calculate_and_plot_cpi(df2, asin_list, start_date, end_date, price_min, price_max, compulsory_features,
#                            same_brand_option):
#     """
#     Perform time-series analysis for ASINs between the given start and end dates.
#     """
#     # df2.to_csv("Serp Data.csv", index=False)
#     asin = asin_list[0]
#     start_date = datetime.strptime(start_date, '%Y-%m-%d')
#     end_date = datetime.strptime(end_date, '%Y-%m-%d')
#     print(start_date)
#     print(end_date)
#
#     all_results = []
#     competitor_count_per_day = []
#     null_price_count_per_day = []
#
#     # Filter df2 by the selected date range
#     current_date = start_date
#     while current_date <= end_date:
#         date_str = current_date.strftime('%Y-%m-%d')
#         df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d')
#         date_str = pd.to_datetime(date_str)
#         df_combined = df2.copy()
#         df_current_day = df_combined[df_combined['date'] == date_str]
#
#         if df_current_day.empty:
#             current_date += timedelta(days=1)
#             continue
#
#         daily_results = []
#         daily_competitor_count = 0
#         daily_null_count = 0
#
#
#         target_price = df_current_day[df_current_day['asin'] == asin]['price'].values[0]
#         #df_current_day.rename(columns={'Title': 'product_title'}, inplace=True)
#         df_current_day.to_csv("Filtered Dataset.csv", index=False)
#         result = run_analysis(asin, price_min, price_max, target_price, compulsory_features, same_brand_option,df_current_day)
#         num_competitors_found = result[3]
#         daily_competitor_count += num_competitors_found
#         daily_results.append((date_str, *result[:-1]))
#
#         # Save competitor details for this date
#         competitor_details_df = result[-1]
#         #competitor_details_df.to_csv(f'competitor_df_{asin}_{date_str}.csv', index=False)
#
#
#
#         # Handle null price values
#         nan_count = df_current_day['price'].isna().sum()
#         zero_count = (df_current_day['price'] == 0).sum()
#         empty_string_count = (df_current_day['price'] == '').sum()
#         daily_null_count = nan_count + zero_count + empty_string_count
#
#         competitor_count_per_day.append(daily_competitor_count)
#         null_price_count_per_day.append(daily_null_count)
#
#         all_results.extend(daily_results)
#         current_date += timedelta(days=1)
#
#     # Create result DataFrame
#     result_df = pd.DataFrame(all_results,
#                              columns=['Date', 'ASIN', 'Target Price', 'CPI Score', 'Number Of Competitors Found',
#                                       'Size', 'Product Dimension', 'Competitor Prices'])
#
#     # Merge with other necessary data for analysis
#     napqueen_df = pd.read_csv("data/ads_data_sep15.csv")
#     napqueen_df['date'] = pd.to_datetime(napqueen_df['date'])
#     napqueen_df = napqueen_df.rename(columns={'date': 'Date', 'asin': 'ASIN'})
#
#     try:
#         result_df['Date'] = pd.to_datetime(result_df['Date'])
#         result_df = pd.merge(result_df, napqueen_df[['Date', 'ASIN', 'ad_spend', 'orderedunits']], on=['Date', 'ASIN'],
#                              how='left')
#
#         print("Merging successful. Here's the result_df info:")
#         print(result_df.info())
#
#     except KeyError as e:
#         print(f"KeyError: {e} - likely missing column during merging.")
#
#     except Exception as e:
#         print(f"An error occurred: {e}")
#
#     # Save analysis results
#     result_df.to_csv(f'data/analysis_results_{start_date.strftime("%Y-%m-%d")}_to_{end_date.strftime("%Y-%m-%d")}.csv',
#                      index=False)
#
#     # Plot the time-series results
#     plot_results(result_df, asin_list, start_date, end_date)

def process_date(df2, asin, date_str, price_min, price_max, compulsory_features, same_brand_option):
    """
    This function processes data for a single date and returns the results.
    """
    df_combined = df2.copy()
    df_combined['date'] = pd.to_datetime(df_combined['date'], format='%Y-%m-%d')
    df_current_day = df_combined[df_combined['date'] == date_str]

    if df_current_day.empty:
        return None

    target_price = df_current_day[df_current_day['asin'] == asin]['price'].values[0]
    result = run_analysis(asin, price_min, price_max, target_price, compulsory_features, same_brand_option, df_current_day)

    daily_null_count = df_current_day['price'].isna().sum() + (df_current_day['price'] == 0).sum() + (df_current_day['price'] == '').sum()

    return {
        'date': date_str,
        'result': result,
        'daily_null_count': daily_null_count,
        'num_competitors_found': result[3],
        'competitors': result[7]
    }


def calculate_and_plot_cpi(df2, asin_list, start_date, end_date, price_min, price_max, compulsory_features,
                           same_brand_option):
    """
    Perform time-series analysis for ASINs between the given start and end dates.
    """
    asin = asin_list[0]
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    all_results = []
    competitor_count_per_day = []
    null_price_count_per_day = []

    current_date = start_date
    dates_to_process = []

    # Prepare list of dates to process
    while current_date <= end_date:
        dates_to_process.append(current_date)
        current_date += timedelta(days=1)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_date, df2, asin, pd.to_datetime(current_date), price_min, price_max, compulsory_features, same_brand_option)
            for current_date in dates_to_process
        ]
        for future in futures:
            result = future.result()
            if result is not None:
                daily_results = result['result'][:-1]
                daily_null_count = result['daily_null_count']
                num_competitors_found = result['num_competitors_found']


                all_results.append((result['date'], *daily_results))
                competitor_count_per_day.append(num_competitors_found)
                null_price_count_per_day.append(daily_null_count)

                # Save competitors data to CSV immediately
                competitor_details_df = result['competitors']
                date_str = result['date'].strftime('%Y-%m-%d')
            
                # Save the CSV file directly in the current directory
                csv_filename = f"competitors_{asin}_{date_str}.csv"
                competitor_details_df.to_csv(csv_filename, index=False)


    # Create result DataFrame
    result_df = pd.DataFrame(all_results,
                             columns=['Date', 'ASIN', 'Target Price', 'CPI Score', 'Number Of Competitors Found',
                                      'Size', 'Product Dimension', 'Competitor Prices','Dynamic CPI'])

    result_df.to_csv("result_df.csv")

    # Merge with other necessary data for analysis
    napqueen_df = pd.read_csv("ads_data_sep15.csv")
    napqueen_df['date'] = pd.to_datetime(napqueen_df['date'], format='%d-%m-%Y', errors='coerce')
    napqueen_df = napqueen_df.rename(columns={'date': 'Date', 'asin': 'ASIN'})

    try:
        result_df['Date'] = pd.to_datetime(result_df['Date'], format='%d-%m-%Y')
        result_df = pd.merge(result_df, napqueen_df[['Date', 'ASIN', 'ad_spend', 'orderedunits']], on=['Date', 'ASIN'],
                             how='left')

        print("Merging successful. Here's the result_df info:")
        print(result_df.info())

    except KeyError as e:
        print(f"KeyError: {e} - likely missing column during merging.")

    except Exception as e:
        print(f"An error occurred: {e}")
    
    # Save analysis results
    result_df.to_csv(f'analysis_results_{start_date.strftime("%Y-%m-%d")}_to_{end_date.strftime("%Y-%m-%d")}.csv',
                     index=False)

    # Plot the time-series results
    plot_results(result_df, asin_list, start_date, end_date)




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
    plt.show()


def plot_results(result_df, asin_list, start_date, end_date):
    plot_window = tk.Toplevel(root)
    plot_window.title("Time-Series Analysis")

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

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


def get_distribution_date(result_df, asin):
    def on_submit():
        selected_date = date_entry.get()
        try:
            selected_date = datetime.strptime(selected_date, '%d-%m-%Y')
            plot_distribution_graph(result_df, asin, selected_date)
            distribution_window.destroy()
        except ValueError:
            messagebox.showerror("Error", "Invalid date format. Please use YYYY-MM-DD.")

    distribution_window = tk.Toplevel(root)
    distribution_window.title("Enter Distribution Date")

    tk.Label(distribution_window, text="Date for Distribution Graph (YYYY-MM-DD):").pack(pady=10)
    date_entry = ttk.Entry(distribution_window)
    date_entry.pack(pady=5)
    submit_button = ttk.Button(distribution_window, text="Submit", command=on_submit)
    submit_button.pack(pady=10)


def plot_distribution_graph(result_df, asin, selected_date):
    asin_results = result_df[result_df['ASIN'] == asin]
    selected_data = asin_results[asin_results['Date'] == selected_date]

    if selected_data.empty:
        messagebox.showerror("Error", "No data available for the selected date.")
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

    fig.show()

def run_analysis_button():
    print("Inside Analysis")
    asin = asin_entry.get().upper()  # Target ASIN

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df_recent = df[df['date'] == df['date'].max()]
    df_recent = df_recent.drop_duplicates(subset=['asin'])

    start_date = start_date_entry.get()
    end_date = end_date_entry.get()

    # Ensure that ASIN exists in the dataset
    if asin not in df['ASIN'].values:
        messagebox.showerror("Error", "ASIN not found.")
        return

    # Extract the product information for the target ASIN
    target_product = df[df['ASIN'] == asin].iloc[0]
    target_brand = target_product['brand'].lower()

    if target_brand is None:
        try:
            target_brand = target_product['Product Details']['Brand'].lower()
        except:
            pass

    print("brand: "+str(target_brand))


    # target_brand = target_product['identified_brand'].lower()

    # Get target price, price range inputs
    try:
        price_min = float(price_min_entry.get())
        price_max = float(price_max_entry.get())
        target_price = float(target_price_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Invalid price range or target price.")
        return

    # Get user options for same brand products and compulsory features
    same_brand_option = same_brand_var.get()
    compulsory_features = [feature for feature, var in compulsory_features_vars.items() if var.get()]

    # Check if we should perform time-series analysis (only if brand == 'napqueen' and dates are provided)
    if target_brand.lower() == "napqueen" and start_date and end_date:
        perform_scatter_plot(asin, target_price, price_min, price_max, compulsory_features, same_brand_option,df_recent)
        calculate_and_plot_cpi(df, [asin], start_date, end_date, price_min, price_max, compulsory_features,same_brand_option)
        #plt.show()
    else:
        # Perform scatter plot only if no dates are provided
        perform_scatter_plot(asin, target_price, price_min, price_max, compulsory_features,same_brand_option,df_recent)  # Scatter plot only


# Load data globally before starting the Tkinter main loop

df = load_and_preprocess_data()

# Initialize Tkinter Window
root = tk.Tk()
root.title("ASIN Competitor Analysis")
root.geometry("500x700")  # Vertical layout with suitable dimensions

main_frame = ttk.Frame(root, padding=(10, 10))
main_frame.pack(fill=tk.BOTH, expand=True)
main_frame.columnconfigure(0, weight=1)
main_frame.columnconfigure(1, weight=4)
main_frame.rowconfigure(0, weight=1)

# Create a scrollable canvas for the input frame
canvas = tk.Canvas(main_frame)
canvas.grid(row=0, column=0, sticky='nsew')

scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
scrollbar.grid(row=0, column=1, sticky='ns')
canvas.configure(yscrollcommand=scrollbar.set)

input_frame = ttk.Frame(canvas)
canvas.create_window((0, 0), window=input_frame, anchor='nw')


def onFrameConfigure(canvas):
    canvas.configure(scrollregion=canvas.bbox("all"))


input_frame.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))

ttk.Label(input_frame, text="ASIN:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
asin_entry = ttk.Entry(input_frame)
asin_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')
ttk.Label(input_frame, text="Price Min:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
price_min_entry = ttk.Entry(input_frame)
price_min_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')
ttk.Label(input_frame, text="Price Max:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
price_max_entry = ttk.Entry(input_frame)
price_max_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')
ttk.Label(input_frame, text="Target Price:").grid(row=3, column=0, padx=5, pady=5, sticky='e')
target_price_entry = ttk.Entry(input_frame)
target_price_entry.grid(row=3, column=1, padx=5, pady=5, sticky='w')
same_brand_var = tk.StringVar(value='all')

# Start Date and End Date entries moved up
ttk.Label(input_frame, text="Start Date (YYYY-MM-DD):").grid(row=4, column=0, padx=5, pady=5, sticky='e')
start_date_entry = ttk.Entry(input_frame)
start_date_entry.grid(row=4, column=1, padx=5, pady=5, sticky='w')

ttk.Label(input_frame, text="End Date (YYYY-MM-DD):").grid(row=5, column=0, padx=5, pady=5, sticky='e')
end_date_entry = ttk.Entry(input_frame)
end_date_entry.grid(row=5, column=1, padx=5, pady=5, sticky='w')

ttk.Radiobutton(input_frame, text="Include all brands", variable=same_brand_var, value='all').grid(row=6, column=0,
                                                                                                   columnspan=2, pady=5)
ttk.Radiobutton(input_frame, text="Show only same brand products", variable=same_brand_var, value='only').grid(row=7,
                                                                                                               column=0,
                                                                                                               columnspan=2,
                                                                                                               pady=5)
ttk.Radiobutton(input_frame, text="Omit same brand products", variable=same_brand_var, value='omit').grid(row=8,
                                                                                                          column=0,
                                                                                                          columnspan=2,
                                                                                                          pady=5)
analyze_button = ttk.Button(input_frame, text="Analyze", command=run_analysis_button)
analyze_button.grid(row=9, column=0, columnspan=2, pady=10)
show_features_button = ttk.Button(input_frame, text="Show Features", command=show_features)
show_features_button.grid(row=10, column=0, columnspan=2, pady=10)

feature_frame_container = ttk.Frame(input_frame)
feature_frame_container.grid(row=11, column=0, columnspan=2, pady=10)
feature_canvas = tk.Canvas(feature_frame_container, height=200)
feature_frame = ttk.Frame(feature_canvas)
scrollbar = ttk.Scrollbar(feature_frame_container, orient="vertical", command=feature_canvas.yview)
feature_canvas.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")
feature_canvas.pack(side="left", fill="both", expand=True)
feature_canvas.create_window((0, 0), window=feature_frame, anchor='nw')


def onFrameConfigure(canvas):
    canvas.configure(scrollregion=canvas.bbox("all"))


feature_frame.bind("<Configure>", lambda event, canvas=feature_canvas: onFrameConfigure(canvas))

product_details_frame = ttk.LabelFrame(input_frame, text="Product Details", padding=(10, 10))
product_details_frame.grid(row=12, column=0, columnspan=2, pady=10, sticky='nsew')

compulsory_features_vars = {}
root.mainloop()