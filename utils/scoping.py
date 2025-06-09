import pandas as pd
import numpy as np
import os

def calculate_materiality(df, benchmark_column_input, benchmark_type, percentage, 
                          account_name_col=None, 
                          specific_mapping_col='Specific_Mapping', 
                          global_mapping_col='Global_Mapping', 
                          balance_col='Balance'):
    """
    Calculate materiality based on a benchmark column and percentage
    
    Parameters:
    - df: DataFrame containing the accounts data
    - benchmark_column_input: User-provided benchmark (column name, account name, or category)
    - benchmark_type: Type of benchmark calculation ('total', 'specific_accounts')
    - percentage: Percentage to apply to benchmark (e.g., 0.5, 1, 2)
    - account_name_col: Name of the original account description column from the input file
    - specific_mapping_col: Name of the column containing specific account mappings
    - global_mapping_col: Name of the column containing global/broad account mappings
    - balance_col: Name of the column containing account balances
    
    Returns:
    - dict: Contains 'benchmark_value', 'materiality', and 'calculation_steps' keys, or 'error' key.
    """
    print(f"\n=== Starting materiality calculation (New Logic) ===")
    print(f"Input Benchmark: {benchmark_column_input}")
    print(f"Benchmark type: {benchmark_type}")
    print(f"Percentage: {percentage}%")
    print(f"Account Name Col: {account_name_col}, Specific Map Col: {specific_mapping_col}, Global Map Col: {global_mapping_col}, Balance Col: {balance_col}")

    calculation_steps = [
        f"Input Benchmark: {benchmark_column_input}",
        f"Benchmark Type: {benchmark_type}",
        f"Percentage: {percentage}%"
    ]
    benchmark_value = 0.0 # Default benchmark value

    try:
        # Convert percentage to decimal and validate
        try:
            percentage_decimal = float(percentage) / 100
            calculation_steps.append(f"Converted percentage to decimal: {percentage}% → {percentage_decimal}")
            print(f"Converted percentage to decimal: {percentage}% → {percentage_decimal}")
        except (ValueError, TypeError) as e:
            error_msg = f"Invalid percentage value: {percentage}"
            calculation_steps.append(f"Error: {error_msg}")
            raise ValueError(error_msg) from e
        
        # Ensure balance column exists and is numeric
        if balance_col not in df.columns:
            error_msg = f"Balance column '{balance_col}' not found in DataFrame."
            calculation_steps.append(f"Error: {error_msg}")
            raise ValueError(error_msg)
        df[balance_col] = pd.to_numeric(df[balance_col], errors='coerce').fillna(0)


        if benchmark_type == 'total':
            calculation_steps.append(f"Processing Benchmark Type: Total")
            # If benchmark_column_input is a direct column name in df (e.g., "Balance")
            if benchmark_column_input in df.columns:
                # Ensure the direct column is numeric if it's not the primary balance_col
                if benchmark_column_input != balance_col:
                     df[benchmark_column_input] = pd.to_numeric(df[benchmark_column_input], errors='coerce').fillna(0)
                benchmark_value = float(df[benchmark_column_input].sum())
                calculation_steps.append(f"Using direct column '{benchmark_column_input}' for sum.")
                print(f"Total of direct column '{benchmark_column_input}': {benchmark_value:,.2f}")
            else:
                # Assume benchmark_column_input is a high-level category (e.g., "Assets", "Income")
                # These typically map to Global_Mapping
                category_to_filter = benchmark_column_input
                calculation_steps.append(f"Interpreting '{benchmark_column_input}' as a category for {global_mapping_col}.")
                if global_mapping_col in df.columns:
                    # Case-insensitive partial match for category
                    filtered_df = df[df[global_mapping_col].astype(str).str.contains(category_to_filter, case=False, na=False)]
                    if not filtered_df.empty:
                        benchmark_value = float(filtered_df[balance_col].sum())
                        calculation_steps.append(f"Filtered by '{global_mapping_col}' containing '{category_to_filter}'. Found {len(filtered_df)} rows.")
                        print(f"Found {len(filtered_df)} rows in '{global_mapping_col}' for category '{category_to_filter}'. Sum of '{balance_col}': {benchmark_value:,.2f}")
                    else:
                        benchmark_value = 0.0
                        calculation_steps.append(f"No rows found in '{global_mapping_col}' containing '{category_to_filter}'. Benchmark set to 0.")
                        print(f"No rows found in '{global_mapping_col}' for category '{category_to_filter}'.")
                else:
                    error_msg = f"'{global_mapping_col}' column not found for 'total' benchmark with category '{benchmark_column_input}'."
                    calculation_steps.append(f"Error: {error_msg}")
                    raise ValueError(error_msg)
        
        elif benchmark_type == 'specific_accounts':
            calculation_steps.append(f"Processing Benchmark Type: Specific Accounts")
            calculation_steps.append(f"Searching for account(s) matching: {benchmark_column_input}")
            found_match = False

            # Priority 1: Specific_Mapping column
            if not found_match and specific_mapping_col in df.columns:
                # Try exact match first (case-insensitive)
                filtered_df = df[df[specific_mapping_col].astype(str).str.lower() == benchmark_column_input.lower()]
                if not filtered_df.empty:
                    benchmark_value = float(filtered_df[balance_col].sum())
                    calculation_steps.append(f"Found {len(filtered_df)} rows by exact match in '{specific_mapping_col}'.")
                    print(f"Exact match in '{specific_mapping_col}' for '{benchmark_column_input}'. Found {len(filtered_df)} rows. Sum: {benchmark_value:,.2f}")
                    found_match = True
                else: # Try contains match (case-insensitive)
                    filtered_df = df[df[specific_mapping_col].astype(str).str.contains(benchmark_column_input, case=False, na=False)]
                    if not filtered_df.empty:
                        benchmark_value = float(filtered_df[balance_col].sum())
                        calculation_steps.append(f"Found {len(filtered_df)} rows by 'contains' match in '{specific_mapping_col}'.")
                        print(f"Contains match in '{specific_mapping_col}' for '{benchmark_column_input}'. Found {len(filtered_df)} rows. Sum: {benchmark_value:,.2f}")
                        found_match = True
                    else:
                        calculation_steps.append(f"No match for '{benchmark_column_input}' in '{specific_mapping_col}'.")
                        print(f"No match for '{benchmark_column_input}' in '{specific_mapping_col}'.")
            
            # Priority 2: Global_Mapping column (if not found in Specific_Mapping)
            if not found_match and global_mapping_col in df.columns:
                # Try exact match first (case-insensitive)
                filtered_df = df[df[global_mapping_col].astype(str).str.lower() == benchmark_column_input.lower()]
                if not filtered_df.empty:
                    benchmark_value = float(filtered_df[balance_col].sum())
                    calculation_steps.append(f"Found {len(filtered_df)} rows by exact match in '{global_mapping_col}'.")
                    print(f"Exact match in '{global_mapping_col}' for '{benchmark_column_input}'. Found {len(filtered_df)} rows. Sum: {benchmark_value:,.2f}")
                    found_match = True
                else: # Try contains match (case-insensitive)
                    filtered_df = df[df[global_mapping_col].astype(str).str.contains(benchmark_column_input, case=False, na=False)]
                    if not filtered_df.empty:
                        benchmark_value = float(filtered_df[balance_col].sum())
                        calculation_steps.append(f"Found {len(filtered_df)} rows by 'contains' match in '{global_mapping_col}'.")
                        print(f"Contains match in '{global_mapping_col}' for '{benchmark_column_input}'. Found {len(filtered_df)} rows. Sum: {benchmark_value:,.2f}")
                        found_match = True
                    else:
                        calculation_steps.append(f"No match for '{benchmark_column_input}' in '{global_mapping_col}'.")
                        print(f"No match for '{benchmark_column_input}' in '{global_mapping_col}'.")

            # Priority 3: Original Account Name Column (if provided and not found elsewhere)
            if not found_match and account_name_col and account_name_col in df.columns:
                # Try exact match first (case-insensitive)
                filtered_df = df[df[account_name_col].astype(str).str.lower() == benchmark_column_input.lower()]
                if not filtered_df.empty:
                    benchmark_value = float(filtered_df[balance_col].sum())
                    calculation_steps.append(f"Found {len(filtered_df)} rows by exact match in original account column '{account_name_col}'.")
                    print(f"Exact match in '{account_name_col}' for '{benchmark_column_input}'. Found {len(filtered_df)} rows. Sum: {benchmark_value:,.2f}")
                    found_match = True
                else: # Try contains match (case-insensitive)
                    filtered_df = df[df[account_name_col].astype(str).str.contains(benchmark_column_input, case=False, na=False)]
                    if not filtered_df.empty:
                        benchmark_value = float(filtered_df[balance_col].sum())
                        calculation_steps.append(f"Found {len(filtered_df)} rows by 'contains' match in original account column '{account_name_col}'.")
                        print(f"Contains match in '{account_name_col}' for '{benchmark_column_input}'. Found {len(filtered_df)} rows. Sum: {benchmark_value:,.2f}")
                        found_match = True
                    else:
                        calculation_steps.append(f"No match for '{benchmark_column_input}' in original account column '{account_name_col}'.")
                        print(f"No match for '{benchmark_column_input}' in original account column '{account_name_col}'.")
            
            if not found_match:
                calculation_steps.append(f"No accounts found matching '{benchmark_column_input}' after checking all relevant columns. Benchmark set to 0.")
                print(f"No accounts found matching '{benchmark_column_input}' after all checks. Benchmark value remains {benchmark_value:,.2f}")
                # benchmark_value is already 0.0 if no matches

        else:
            error_msg = f"Unknown benchmark type: {benchmark_type}"
            calculation_steps.append(f"Error: {error_msg}")
            raise ValueError(error_msg)
        
        calculation_steps.append(f"Final Benchmark Value: {benchmark_value:,.2f}")
        print(f"Final Benchmark Value: {benchmark_value:,.2f}")
        
        # Calculate materiality
        materiality = benchmark_value * percentage_decimal
        calculation_steps.append(f"Calculated Materiality: {benchmark_value:,.2f} × {percentage_decimal} = {materiality:,.2f}")
        print(f"Calculated materiality: {benchmark_value:,.2f} × {percentage_decimal} = {materiality:,.2f}")
        
        result = {
            'benchmark_value': benchmark_value,
            'overall_materiality': materiality,  # Renamed key
            'calculation_steps': calculation_steps # Add detailed steps to result
        }
        
        print("=== Materiality calculation (New Logic) completed successfully ===")
        return result
        
    except Exception as e:
        error_msg = f"Error in materiality calculation (New Logic): {str(e)}"
        calculation_steps.append(f"Runtime Error: {error_msg}") # Add error to steps if it occurs mid-way
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        # Ensure calculation_steps is part of the error dict if it was populated
        return {'error': error_msg, 'calculation_steps': calculation_steps if 'calculation_steps' in locals() else [f"Error: {error_msg}"]}

def calculate_performance_materiality(materiality, pm_percentage=75):
    """
    Calculate performance materiality based on materiality
    
    Parameters:
    - materiality: The calculated materiality value
    - pm_percentage: Percentage of materiality to use (typically 75-80%)
    
    Returns:
    - dict: Contains 'performance_materiality' and 'performance_materiality_percentage' keys, or 'error' key.
    """
    try:
        pm_percentage_decimal = float(pm_percentage) / 100
        performance_materiality_value = materiality * pm_percentage_decimal
        print(f"Performance Materiality ({pm_percentage}% of {materiality:,.2f}): {performance_materiality_value:,.2f}")
        return {
            'performance_materiality': performance_materiality_value,
            'performance_materiality_percentage': pm_percentage
        }
    except (ValueError, TypeError) as e:
        error_msg = f"Error calculating performance materiality: Invalid input. Materiality: {materiality}, PM Percentage: {pm_percentage}. Error: {e}"
        print(error_msg)
        return {'error': error_msg, 'performance_materiality': None, 'performance_materiality_percentage': pm_percentage}
    except Exception as e:
        error_msg = f"An unexpected error occurred in calculate_performance_materiality: {e}"
        print(error_msg)
        return {'error': error_msg, 'performance_materiality': None, 'performance_materiality_percentage': pm_percentage}

def scope_accounts(df, performance_materiality, balance_column='Balance'):
    """
    Determine which accounts are in scope based on performance materiality
    
    Parameters:
    - df: DataFrame containing the accounts data
    - performance_materiality: The calculated performance materiality value
    - balance_column: Column containing account balances
    
    Returns:
    - DataFrame with additional columns for scoping
    """
    try:
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Ensure numeric balance column
        result_df[balance_column] = pd.to_numeric(result_df[balance_column], errors='coerce').fillna(0)
        
        # Create absolute balance column for comparison (since we care about magnitude, not sign)
        result_df['Abs_Balance'] = result_df[balance_column].abs()
        
        # Determine quantitative scoping based on performance materiality
        result_df['Quantitative_Scope'] = np.where(
            result_df['Abs_Balance'] >= performance_materiality,
            'In Scope',
            'Out of Scope'
        )
        
        # Add column for qualitative scoping decisions (defaults to same as quantitative)
        result_df['Qualitative_Scope'] = result_df['Quantitative_Scope']
        
        # Always include cash accounts in scope (example of qualitative override)
        if 'Specific_Mapping' in result_df.columns:
            cash_mask = result_df['Specific_Mapping'].str.contains('Cash', case=False, na=False)
            result_df.loc[cash_mask, 'Qualitative_Scope'] = 'In Scope'
        
        # Final scope decision (if either quantitative or qualitative is in scope)
        result_df['Final_Scope'] = np.where(
            (result_df['Quantitative_Scope'] == 'In Scope') | 
            (result_df['Qualitative_Scope'] == 'In Scope'),
            'In Scope',
            'Out of Scope'
        )
        
        # Add justification column for qualitative decisions
        result_df['Scope_Justification'] = np.where(
            (result_df['Qualitative_Scope'] == 'In Scope') & (result_df['Quantitative_Scope'] == 'Out of Scope'),
            'Qualitative override',
            ''
        )
        
        return result_df
    
    except Exception as e:
        raise Exception(f"Error in scoping accounts: {str(e)}")

def save_scoped_excel(df, output_path):
    """
    Save the scoped DataFrame to Excel
    """
    try:
        # Ensure all required columns exist
        required_cols = ['Quantitative_Scope', 'Qualitative_Scope', 'Final_Scope', 'Scope_Justification']
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
    except Exception as e:
        raise Exception(f"Error saving scoped Excel file: {str(e)}")
