import pandas as pd
import numpy as np

def clean_dataset():
    # Read the Excel file
    try:
        df = pd.read_excel('products.xlsx')
        print(f"Original dataset shape: {df.shape}")
        
        # Display initial missing values
        print("\nMissing values before cleaning:")
        print(df.isnull().sum())
        
        # Clean text columns
        df['title'] = df['title'].fillna('')
        df['brand'] = df['brand'].fillna('')
        df['description'] = df['description'].fillna('')
        
        # Clean price column
        def clean_price(value):
            if pd.isna(value):
                return 0.00
            try:
                if isinstance(value, str):
                    return float(value.replace('$', '').replace(',', '').strip())
                return float(value)
            except:
                return 0.00
                
        df['price/value'] = df['price/value'].apply(clean_price).round(2)
        
        # Clean stars column
        def clean_stars(value):
            if pd.isna(value):
                return 0
            try:
                return int(float(str(value)))
            except:
                return 0
                
        df['stars'] = df['stars'].apply(clean_stars)
        
        # Clean reviews count
        def clean_reviews(value):
            if pd.isna(value):
                return 0
            try:
                if isinstance(value, str):
                    return int(value.replace(',', ''))
                return int(value)
            except:
                return 0
                
        df['reviewsCount'] = df['reviewsCount'].apply(clean_reviews)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['title', 'brand'], keep='first')
        
        # Display final missing values
        print("\nMissing values after cleaning:")
        print(df.isnull().sum())
        
        # Display sample of cleaned data
        print("\nSample of cleaned data:")
        print(df.head())
        
        # Save cleaned dataset
        df.to_excel('products_cleaned.xlsx', index=False)
        print(f"\nFinal dataset shape: {df.shape}")
        print("Cleaned data saved to 'products_cleaned.xlsx'")
        
        return df
        
    except Exception as e:
        print(f"Error during cleaning: {e}")
        return None

if __name__ == "__main__":
    cleaned_df = clean_dataset()