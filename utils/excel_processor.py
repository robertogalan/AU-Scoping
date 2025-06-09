import pandas as pd
import os

def find_header_row(filepath, ext, sample_size=10):
    """
    Find the row that contains the actual header in the spreadsheet.
    Looks for the first row with a significant number of non-empty, non-numeric values.
    """
    if ext.lower() == '.csv':
        # For CSV, try to detect if header is in first row
        df_sample = pd.read_csv(filepath, nrows=sample_size, header=None, dtype=str)
    else:
        # For Excel, check multiple rows
        df_sample = pd.read_excel(filepath, header=None, nrows=sample_size, dtype=str)
    
    # Find first row with mostly non-numeric, non-empty values
    for i in range(min(10, len(df_sample))):
        row = df_sample.iloc[i].dropna()
        if len(row) > 0:
            # Count non-numeric values in the row
            non_numeric = sum(not str(val).replace('.', '').replace('-', '').strip().isdigit() 
                            and str(val).strip() != '' for val in row)
            if non_numeric / len(row) > 0.5:  # More than 50% non-numeric values
                return i
    
    return 0  # Default to first row if no good header found

def process_excel_file(filepath):
    """
    Process the uploaded Excel or CSV file into a pandas DataFrame.
    Handles cases where headers are not in the first row and detects data types.
    """
    _, ext = os.path.splitext(filepath)
    
    try:
        # First, try to find the header row
        header_row = find_header_row(filepath, ext)
        
        # Read the file with the detected header row
        if ext.lower() == '.csv':
            df = pd.read_csv(filepath, header=header_row, dtype=str, na_values=['', 'NA', 'N/A', 'NaN'])
        else:  # .xlsx or .xls
            df = pd.read_excel(filepath, header=header_row, dtype=str, na_values=['', 'NA', 'N/A', 'NaN'])
        
        # Clean up column names - strip whitespace and handle empty column names
        df.columns = [str(col).strip() if pd.notna(col) and str(col).strip() != '' 
                     else f'Column_{i}' for i, col in enumerate(df.columns, 1)]
        
        # Convert empty strings to NaN and then drop completely empty rows/columns
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Convert numeric columns to appropriate types
        for col in df.columns:
            # Skip if all values are empty
            if df[col].isna().all():
                continue
                
            # Try to convert to numeric, if possible
            try:
                numeric_vals = pd.to_numeric(df[col], errors='coerce')
                if not numeric_vals.isna().all():  # If at least some values are numeric
                    df[col] = numeric_vals
            except (ValueError, TypeError):
                pass
        
        return df
        
    except Exception as e:
        # Fallback to simple read if anything goes wrong
        print(f"Error processing file with advanced method: {str(e)}")
        if ext.lower() == '.csv':
            df = pd.read_csv(filepath, dtype=str, na_values=['', 'NA', 'N/A', 'NaN'])
        else:
            df = pd.read_excel(filepath, dtype=str, na_values=['', 'NA', 'N/A', 'NaN'])
        
        # Basic cleanup
        df = df.dropna(how='all').dropna(axis=1, how='all')
        df.columns = [str(col).strip() if pd.notna(col) and str(col).strip() != '' 
                     else f'Column_{i}' for i, col in enumerate(df.columns, 1)]
        
        return df

def save_mapped_excel(df, output_path):
    """
    Save the processed DataFrame to Excel with mapping columns
    """
    # Ensure required mapping columns exist
    required_cols = ['Global_Mapping', 'Specific_Mapping']
    for col in required_cols:
        if col not in df.columns:
            df[col] = ''
    
    # Save the file
    _, ext = os.path.splitext(output_path)
    
    if ext.lower() == '.csv':
        df.to_csv(output_path, index=False)
    else:  # .xlsx or .xls
        df.to_excel(output_path, index=False)
    
    return output_path

def calculate_balance(row, balance_cols):
    """
    Calculate the final balance based on debit/credit columns
    """
    if 'balance' in balance_cols:
        # If there's a single balance column
        return row[balance_cols['balance']] if pd.notna(row[balance_cols['balance']]) else 0
    else:
        # If there are separate debit/credit columns
        debit = row[balance_cols['debit']] if pd.notna(row[balance_cols['debit']]) else 0
        credit = row[balance_cols['credit']] if pd.notna(row[balance_cols['credit']]) else 0
        return debit - credit  # Debit is positive, credit is negative in accounting
