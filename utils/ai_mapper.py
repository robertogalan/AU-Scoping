import json
import os
import re
import time
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from openai import OpenAI
from fuzzywuzzy import fuzz, process
from collections import defaultdict

# Initialize OpenAI client with environment variable
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Model configuration
AI_MODEL = "gpt-4.1-mini"  # Using the specified model

# Global mapping patterns for fuzzy matching 
GLOBAL_MAPPING_PATTERNS = {
    'Asset': [
        # Current Assets
        'cash', 'bank', 'checking', 'savings', 'money market', 'petty cash', 'cash equivalents',
        'short term investments', 'marketable securities', 'trading securities', 'commercial paper',
        'account receivable', 'accounts receivable', 'ar ', 'receivable', 'trade receivable', 'debtor', 'customer',
        'allowance for doubtful accounts', 'bad debt reserve',
        'inventory', 'stock', 'merchandise', 'goods', 'supplies', 'raw materials', 'work in progress', 'finished goods',
        'prepaid', 'pre-paid', 'pre pay', 'prepay', 'prepaid expense', 'prepaid insurance', 'prepaid rent',
        'deposit', 'security deposit', 'rent deposit', 'utility deposit',
        'other current asset', 'short term asset', 'current portion of notes receivable',
        
        # Long-term Investments
        'long term investments', 'bonds payable', 'held to maturity securities', 'available for sale securities',
        'equity investments', 'investment in affiliates', 'investment in subsidiary', 'investment property',
        'sinking fund', 'pension fund', 'endowment fund',
        
        # Property, Plant and Equipment
        'fixed asset', 'ppe', 'property', 'plant', 'equipment', 'land', 'building', 'leasehold improvements',
        'furniture', 'fixture', 'machinery', 'vehicles', 'computer equipment', 'office equipment',
        'construction in progress', 'accumulated depreciation', 'accumulated amortization',
        'less accumulated depreciation', 'net property plant equipment',
        
        # Intangible Assets
        'intangible', 'goodwill', 'patent', 'trademark', 'copyright', 'license', 'franchise', 'brand', 
        'intellectual property', 'software', 'website', 'domain name', 'customer list', 'non compete',
        'research and development', 'organization costs', 'startup costs',
        
        # Other Assets
        'deferred tax asset', 'deferred income tax', 'prepaid pension cost', 'bond issue costs',
        'long term notes receivable', 'long term deposits', 'other noncurrent asset',
        'other long term asset', 'other asset'
    ],
    'Liability': [
        # Current Liabilities
        'account payable', 'accounts payable', 'ap ', 'trade payable', 'payable', 'creditor', 'vendor', 'supplier',
        'accrued', 'accrual', 'accrued expense', 'accrued liability', 'accrued payroll', 'accrued wages',
        'accrued interest', 'accrued taxes', 'accrued vacation', 'accrued bonus', 'accrued commission',
        'provision', 'warranty provision', 'provision for returns', 'provision for litigation',
        'short term loan', 'current portion of long term debt', 'current maturities of long term debt',
        'bank overdraft', 'line of credit', 'credit card payable', 'customer deposit', 'unearned revenue',
        'deferred revenue', 'unearned income', 'advance payment', 'customer prepayment',
        'income tax payable', 'payroll tax payable', 'sales tax payable', 'vat payable', 'withholding tax',
        'dividend payable', 'dividends payable', 'interest payable', 'bonus payable',
        'other current liability', 'short term liability',
        
        # Long-term Liabilities
        'long term debt', 'notes payable', 'bonds payable', 'mortgage payable', 'lease liability',
        'capital lease obligation', 'finance lease liability', 'pension liability', 'pbo',
        'deferred tax liability', 'deferred income tax', 'deferred compensation',
        'other long term liability', 'long term liability',
        
        # Other Liabilities
        'warranty liability', 'litigation reserve', 'restructuring reserve', 'environmental liability',
        'asset retirement obligation', 'derivative liability', 'contingent liability'
    ],
    'Equity': [
        # Owner's/Shareholders' Equity
        'common stock', 'preferred stock', 'treasury stock', 'treasury shares', 'additional paid in capital',
        'apic', 'paid in capital', 'share capital', 'issued capital', 'common shares', 'preferred shares',
        'share premium', 'capital surplus', 'contributed surplus', 'donated capital',
        'retained earnings', 'retained deficit', 'accumulated deficit', 'earned surplus',
        'dividend', 'dividends', 'dividend payable', 'dividends declared', 'dividend distribution',
        'drawing', 'withdrawal', 'owner draw', 'partner draw', 'member draw',
        'other comprehensive income', 'oci', 'aoci', 'foreign currency translation',
        'unrealized gain loss', 'revaluation surplus', 'hedging reserve',
        'non controlling interest', 'minority interest', 'nci'
    ],
    'Income': [
        # Operating Revenue
        'revenue', 'sale', 'sales', 'income', 'fees earned', 'service revenue', 'service income',
        'product sales', 'merchandise sales', 'gross sales', 'net sales', 'sales returns', 'sales allowances',
        'sales discount', 'trade discount', 'volume discount', 'rebate', 'refund',
        'royalty', 'license fee', 'franchise fee', 'subscription', 'membership',
        'rental income', 'lease income', 'interest income', 'dividend income',
        'commission', 'brokerage', 'consulting fee', 'professional fee',
        'other income', 'miscellaneous income', 'sundry income',
        
        # Other Income
        'gain', 'gain on sale', 'profit on sale', 'capital gain', 'foreign exchange gain',
        'interest revenue', 'investment income', 'dividend revenue', 'rent revenue',
        'bad debt recovery', 'insurance claim', 'lawsuit settlement', 'government grant',
        'subsidy', 'rebate income', 'scrap sales', 'sale of assets', 'sale of fixed assets'
    ],
    'Expense': [
        # Cost of Goods Sold (COGS)
        'cost of goods sold', 'cogs', 'cost of sales', 'cost of revenue', 'cost of services',
        'direct materials', 'direct labor', 'direct expenses', 'manufacturing overhead',
        'purchase', 'purchases', 'inventory purchases', 'raw materials', 'work in process',
        'freight in', 'shipping in', 'duty', 'tariff', 'import duty', 'customs duty',
        'discount received', 'purchase discount', 'purchase return', 'purchase allowance',
        
        # Operating Expenses
        'salary', 'wage', 'payroll', 'compensation', 'benefit', 'bonus', 'commission',
        'payroll tax', 'payroll expense', 'employee benefit', 'health insurance',
        'pension expense', 'retirement plan', '401k', 'profit sharing', 'stock option',
        'payroll processing', 'payroll service', 'workers comp', 'unemployment tax',
        
        # Rent and Occupancy
        'rent', 'lease', 'rent expense', 'lease expense', 'building rent', 'office rent',
        'warehouse rent', 'equipment rent', 'equipment lease', 'vehicle lease',
        'property tax', 'real estate tax', 'property insurance', 'building insurance',
        'common area maintenance', 'cam', 'property management', 'facility expense',
        
        # Utilities
        'utility', 'utilities', 'electricity', 'power', 'water', 'sewer', 'gas', 'natural gas',
        'internet', 'broadband', 'telephone', 'phone', 'mobile', 'cell phone', 'cable',
        'trash', 'garbage', 'recycling', 'waste disposal', 'snow removal', 'landscaping',
        
        # Insurance
        'insurance', 'premium', 'coverage', 'liability insurance', 'general liability',
        'professional liability', 'errors and omissions', 'e&o', 'directors and officers',
        'd&o', 'business insurance', 'commercial insurance', 'business interruption',
        'cyber liability', 'data breach', 'umbrella policy',
        
        # Travel and Entertainment
        'travel', 't&e', 'transportation', 'airfare', 'lodging', 'hotel', 'meal', 'meals',
        'entertainment', 'client entertainment', 'business meal', 'per diem', 'mileage',
        'car rental', 'taxi', 'uber', 'lyft', 'parking', 'toll', 'subway', 'transit',
        
        # Office and Administrative
        'office supply', 'stationery', 'printing', 'postage', 'shipping', 'delivery',
        'courier', 'freight', 'delivery expense', 'shipping expense', 'freight out',
        'postage expense', 'mailing', 'office expense', 'general expense',
        'bank charge', 'bank fee', 'transaction fee', 'merchant fee', 'credit card fee',
        'payment processing', 'ach fee', 'wire transfer', 'nsf fee', 'overdraft fee',
        
        # Professional Services
        'professional fee', 'legal', 'accounting', 'audit', 'consulting', 'advisory',
        'bookkeeping', 'tax preparation', 'tax filing', 'tax advisory', 'financial advisory',
        'investment advisory', 'management fee', 'franchise fee', 'royalty expense',
        'subscription', 'membership', 'dues', 'license', 'software', 'saas', 'cloud',
        
        # Marketing and Advertising
        'marketing', 'advertising', 'promotion', 'branding', 'public relation', 'pr',
        'social media', 'seo', 'sem', 'ppc', 'google ads', 'facebook ads', 'online advertising',
        'print advertising', 'tv', 'radio', 'billboard', 'direct mail', 'brochure', 'catalog',
        'trade show', 'exhibition', 'sponsorship', 'charity', 'donation', 'contribution',
        
        # Repairs and Maintenance
        'repair', 'maintenance', 'r&m', 'upkeep', 'service contract', 'janitorial', 'cleaning',
        'pest control', 'security', 'alarm', 'surveillance', 'cctv', 'locksmith',
        'landscaping', 'gardening', 'snow removal', 'pool service', 'hvac', 'plumbing',
        'electrical', 'handyman', 'general contractor', 'building maintenance',
        
        # Training and Development
        'training', 'education', 'development', 'seminar', 'workshop', 'conference',
        'certification', 'license renewal', 'professional development', 'tuition reimbursement',
        'employee education', 'training materials', 'online course', 'webinar',
        
        # Technology and Software
        'computer', 'software', 'hardware', 'it', 'information technology', 'tech support',
        'hosting', 'website', 'domain', 'ssl', 'cybersecurity', 'data backup', 'cloud storage',
        'phone system', 'voip', 'internet service', 'isp', 'cable', 'dsl', 'fiber',
        'equipment rental', 'equipment lease', 'equipment maintenance',
        
        # Taxes and Licenses
        'tax', 'taxes', 'income tax', 'payroll tax', 'sales tax', 'vat', 'gst', 'hst',
        'property tax', 'business tax', 'franchise tax', 'excise tax', 'use tax',
        'business license', 'permit', 'registration', 'filing fee', 'annual report',
        'corporate filing', 'franchise fee', 'royalty',
        
        # Depreciation and Amortization
        'depreciation', 'amortization', 'depletion', 'd&a', 'impairment', 'write down',
        'write off', 'bad debt', 'doubtful account', 'allowance for bad debt',
        'inventory obsolescence', 'inventory write down', 'asset disposal',
        
        # Other Expenses
        'interest', 'interest expense', 'loan interest', 'credit card interest', 'bank charges',
        'finance charge', 'late fee', 'penalty', 'fine', 'legal settlement', 'lawsuit',
        'insurance claim', 'casualty loss', 'theft', 'embezzlement', 'fraud',
        'restructuring', 'reorganization', 'severance', 'layoff', 'termination',
        'recruiting', 'hiring', 'job board', 'background check', 'drug test',
        'employee recognition', 'award', 'gift', 'prize', 'incentive',
        'office party', 'holiday party', 'team building', 'employee appreciation',
        'miscellaneous', 'sundry', 'other expense', 'extraordinary item', 'unusual item'
    ]
}

SPECIFIC_MAPPING_PATTERNS = {
    'Asset': {
        'Cash and Cash Equivalents': ['cash', 'bank', 'money market', 'petty cash'],
        'Accounts Receivable': ['account receivable', 'trade receivable', 'debtor'],
        'Inventory': ['inventory', 'stock', 'merchandise', 'goods'],
        'Fixed Assets': ['fixed asset', 'property', 'plant', 'equipment', 'ppe'],
        'Investments': ['investment', 'security', 'bond', 'stock holding']
    },
    'Liability': {
        'Accounts Payable': ['account payable', 'trade payable', 'creditor'],
        'Accrued Expenses': ['accrued', 'accrual', 'provision'],
        'Loans Payable': ['loan', 'borrowing', 'credit', 'debt'],
        'Deferred Revenue': ['deferred revenue', 'unearned revenue', 'advance payment']
    },
    'Equity': {
        'Common Stock': ['common stock', 'share capital', 'issued capital'],
        'Retained Earnings': ['retained earning', 'accumulated profit', 'retained profit'],
        'Additional Paid-in Capital': ['additional paid', 'share premium', 'paid in capital']
    },
    'Income': {
        'Sales Revenue': ['sales', 'revenue', 'income', 'service revenue'],
        'Interest Income': ['interest income', 'interest earned'],
        'Other Income': ['other income', 'miscellaneous income', 'gain']
    },
    'Expense': {
        'Cost of Goods Sold': ['cost of goods', 'cogs', 'cost of sales', 'inventory expense'],
        'Cost of Services': ['cost of services', 'service cost', 'cost of service'],
        'Salary and Wages': ['salary', 'wage', 'payroll', 'compensation', 'employee benefits', 'payroll expense'],
        'Rent and Occupancy': ['rent', 'lease', 'facility', 'occupancy', 'office space'],
        'Depreciation and Amortization': ['depreciation', 'amortization', 'd&a'],
        'Professional Services': ['professional fees', 'legal', 'accounting', 'consulting', 'professional service'],
        'Marketing and Advertising': ['marketing', 'advertising', 'promotion', 'branding'],
        'Technology and Software': ['software', 'technology', 'it services', 'hosting', 'cloud'],
        'General and Administrative': ['general', 'administrative', 'office', 'overhead', 'g&a'],
        'Travel and Entertainment': ['travel', 'entertainment', 'business meal', 't&e']
    }
}

def identify_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Use AI to identify account name, account number, and balance columns in the dataframe.
    This is the first step in the process - the AI makes the initial column identification,
    and the human can review and correct if needed.
    """
    if df.empty:
        return {
            "account_name_col": None,
            "account_number_col": None,
            "balance_type": "combined",
            "balance_cols": {"balance": None}
        }
    
    # First try AI-based identification
    ai_mapping = _ai_identify_columns(df)
    
    # If AI failed to identify required columns, use fallback
    if not ai_mapping.get('account_name_col') or not ai_mapping.get('balance_cols'):
        return fallback_column_identification(df)
    
    return ai_mapping

def _ai_identify_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """Internal function that uses AI to identify columns"""
    try:
        # Get sample data for AI to analyze
        sample_data = df.head(5).to_dict('records')
        sample_json = json.dumps(sample_data, default=str)  # Handle non-serializable types
        columns = df.columns.tolist()
        
        # Analyze data types for each column
        column_analysis = {}
        for col in columns:
            col_series = df[col].dropna()
            sample_values = col_series.head(3).tolist()
            column_analysis[col] = {
                'dtype': str(df[col].dtype),
                'sample_values': sample_values,
                'unique_count': len(col_series.unique()),
                'is_numeric': pd.api.types.is_numeric_dtype(df[col])
            }
        
        # Call OpenAI API to identify columns
        try:
            # First try the chat completions API
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """You are an AI assistant specialized in accounting data analysis. 
                    Your task is to identify key columns in an accounting dataset. 
                    Focus on finding account names, account numbers, and balance/debit/credit columns."""},
                    {"role": "user", "content": f"""Analyze this accounting data and identify the key columns.

COLUMNS AND THEIR PROPERTIES:
{json.dumps(column_analysis, indent=2)}

SAMPLE DATA (first 5 rows):
{sample_json}

INSTRUCTIONS:
1. First, try to identify the account name column (usually contains descriptive names like 'Cash', 'Accounts Receivable')
2. Look for an account number column (often contains codes like '1000', '1100')
3. Check for balance columns - these could be:
   - A single 'balance' column (positive/negative or DR/CR)
   - Separate 'debit' and 'credit' columns
   - Columns with amounts that can be positive (credit) or negative (debit)

RETURN FORMAT (must be valid JSON):
{{
  "account_name_col": "name of the account name column",
  "account_number_col": "name of the account number column (or null if not found)",
  "balance_type": "combined" or "separate",
  "balance_cols": {{
    "balance": "name of balance column"  // if balance_type is "combined"
    // OR
    "debit": "name of debit column",     // if balance_type is "separate"
    "credit": "name of credit column"    // if balance_type is "separate"
  }}
}}

IMPORTANT: 
- Only include columns that you are highly confident about. 
- If unsure about a column, set it to null.
- Respond with ONLY the JSON object, no other text or explanation."""}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Extract the response content
            ai_response = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = ai_response[json_start:json_end]
                column_mapping = json.loads(json_str)
                
                # Validate the response has the required structure
                if not isinstance(column_mapping, dict):
                    raise ValueError("Invalid response format - expected a dictionary")
                
                # Ensure required fields exist with proper defaults
                result = {
                    'account_name_col': column_mapping.get('account_name_col'),
                    'account_number_col': column_mapping.get('account_number_col'),
                    'balance_type': column_mapping.get('balance_type', 'combined'),
                    'balance_cols': column_mapping.get('balance_cols', {})
                }
                
                # Validate balance_cols based on balance_type
                if result['balance_type'] == 'combined':
                    if not result['balance_cols'].get('balance'):
                        result['balance_cols'] = {}
                elif result['balance_type'] == 'separate':
                    if not (result['balance_cols'].get('debit') and result['balance_cols'].get('credit')):
                        result['balance_cols'] = {}
                
                return result
            
            raise ValueError("No valid JSON found in AI response")
            
        except Exception as e:
            print(f"Error in AI column identification: {str(e)}")
            raise Exception(f"AI column identification failed: {str(e)}")
            
    except Exception as e:
        error_msg = str(e)
        print(f"Error in AI column identification: {error_msg}")
        # If we get here, AI identification failed
        if "maximum context length" in error_msg.lower():
            raise Exception("The file is too large for AI processing. Please try with a smaller file or fewer columns.")
        elif "rate limit" in error_msg.lower():
            raise Exception("Rate limit exceeded. Please try again in a moment.")
        else:
            raise Exception(f"AI column identification failed: {error_msg}")

def is_numeric_column(series: pd.Series, sample_size: int = 50) -> bool:
    """Check if a column contains numeric data"""
    # Check if the column has any non-null values
    if series.isna().all():
        return False
        
    # Sample data for faster processing
    sample = series.dropna().sample(min(sample_size, len(series)))
    
    # Check if values are numeric
    try:
        numeric_vals = pd.to_numeric(sample, errors='coerce')
        # Consider it numeric if at least 80% of non-null values can be converted to numbers
        return numeric_vals.notna().mean() > 0.8
    except (ValueError, TypeError):
        return False

def analyze_columns_for_balances(df: pd.DataFrame) -> tuple:
    """Analyze columns to identify potential balance columns"""
    numeric_cols = []
    potential_balances = []
    
    for col in df.columns:
        if is_numeric_column(df[col]):
            numeric_cols.append(col)
            
            # Check if column name suggests it's a balance/amount
            balance_terms = ['balance', 'amount', 'total', 'value', 'amt', 'bal', 'dr', 'cr', 'debit', 'credit']
            if any(term in str(col).lower() for term in balance_terms):
                potential_balances.append(col)
    
    return numeric_cols, potential_balances

def fallback_column_identification(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Enhanced method to identify columns by analyzing both column names and data content.
    Returns a dictionary with identified columns for account names, numbers, and balances.
    """
    if df.empty or len(df.columns) == 0:
        return {
            "account_name_col": None,
            "account_number_col": None,
            "balance_type": "combined",
            "balance_cols": {"balance": None}
        }
    
    columns = df.columns.tolist()
    column_mapping = {
        "account_name_col": None,
        "account_number_col": None,
        "balance_type": "combined",
        "balance_cols": {"balance": None}
    }
    
    # Analyze numeric columns for potential balances
    numeric_cols, potential_balances = analyze_columns_for_balances(df)
    
    # Look for account name column
    name_patterns = ["name", "account name", "description", "account", "title", "label"]
    name_cols = []
    
    for pattern in name_patterns:
        matches = [col for col in columns if pattern.lower() in str(col).lower()]
        for match in matches:
            if match not in name_cols:
                name_cols.append(match)
    
    # If we found potential name columns, use the first one that's not numeric
    for col in name_cols:
        if not is_numeric_column(df[col]):
            column_mapping["account_name_col"] = col
            break
    
    # If no name column found by name, try to find a non-numeric column with text data
    if not column_mapping["account_name_col"]:
        for col in df.columns:
            if not is_numeric_column(df[col]) and df[col].nunique() > 1:
                column_mapping["account_name_col"] = col
                break
    
    def is_likely_account_number(series: pd.Series, sample_size: int = 100) -> bool:
        """Check if a column is likely to contain account numbers"""
        if series.empty or series.dtype != 'object':
            return False
            
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
            
        # Adjust sample size to not exceed population
        sample_size = min(sample_size, len(non_null))
        if sample_size == 0:
            return False
            
        try:
            sample = non_null.sample(sample_size, replace=False)
            
            # Check for common account number patterns
            pattern_matches = sample.astype(str).str.match(r'^\s*\d{2,6}(\s*-\s*[A-Za-z].*)?$')
            if not pattern_matches.empty and pattern_matches.any():
                # Check if most values match the pattern
                return pattern_matches.mean() > 0.7  # At least 70% match the pattern
        except Exception as e:
            print(f"Error in is_likely_account_number: {str(e)}")
            return False
            
        return False
    
    # Look for account number column by name first
    number_patterns = [
        "account number", "account no", "account #", "acct #", 
        "acct no", "acct num", "account code", "acct code"
    ]
    
    for pattern in number_patterns:
        matches = [col for col in columns 
                 if pattern.lower() in str(col).lower() 
                 and col != column_mapping["account_name_col"]]
        if matches:
            column_mapping["account_number_col"] = matches[0]
            break
    
    # If no number column found by name, look for columns with account number patterns
    if not column_mapping["account_number_col"] and len(df) > 10:
        # Skip columns that are already identified as balances or have balance-related names
        balance_keywords = ['balance', 'amount', 'debit', 'credit', 'value', 'total']
        
        for col in df.columns:
            col_lower = str(col).lower()
            # Skip if column is already used or has balance-related keywords
            if (col == column_mapping["account_name_col"] or
                any(kw in col_lower for kw in balance_keywords)):
                continue
                
            # Check if column contains account numbers
            if is_likely_account_number(df[col]):
                column_mapping["account_number_col"] = col
                break
    
    # As a last resort, look for a column with mostly unique numeric values
    if not column_mapping["account_number_col"] and len(df) > 10:
        for col in df.select_dtypes(include=['number']).columns:
            if (col != column_mapping["account_name_col"] and 
                df[col].nunique() / len(df) > 0.7 and  # High uniqueness
                not any(kw in str(col).lower() for kw in balance_keywords)):  # Not a balance column
                column_mapping["account_number_col"] = col
                break
    
    def is_likely_balance_column(series: pd.Series, col_name: str) -> bool:
        """Check if a column is likely to contain balance amounts"""
        # Skip non-numeric columns
        if not is_numeric_column(series):
            return False
            
        col_name = str(col_name).lower()
        
        # Check for balance-related keywords in column name
        balance_keywords = [
            'balance', 'amount', 'value', 'total', 'amt', 'bal', 'sum', 'net',
            'ending balance', 'current balance', 'beginning balance',
            'balance forward', 'balance b/f', r'balance b\d'  # Using raw string for regex pattern
        ]
        
        # Check for debit/credit indicators in column name
        if any(kw in col_name for kw in ['debit', 'dr', 'credit', 'cr']):
            return True
            
        # If column name contains balance-related keywords, it's likely a balance column
        if any(kw in col_name for kw in balance_keywords):
            return True
            
        # Check if values look like monetary amounts
        sample = series.dropna().sample(min(100, len(series))).astype(str)
        money_pattern = r'^[\$€£]?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?$|^\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*[\$€£]?$'
        money_matches = sample.str.match(money_pattern, na=False)
        
        if money_matches.any() and money_matches.mean() > 0.8:  # At least 80% match money pattern
            return True
            
        return False
    
    # First, look for balance columns - prioritize this over account number identification
    # as balance columns are more critical for the mapping
    balance_patterns = [
        # Most specific patterns first
        'sum of balance', 'ending balance', 'current balance', 'beginning balance',
        'balance forward', 'balance b/f', 'balance b\\d',
        # General balance terms
        'balance', 'amount', 'total', 'value', 'amt', 'bal', 'sum', 'net',
        # Currency symbols
        '$', '€', '£', '¥', '₹', '₽',
        # Accounting terms
        'assets', 'liabilities', 'equity', 'revenue', 'expense',
        'income', 'profit', 'loss', 'debit', 'credit', 'dr', 'cr'
    ]
    
    # Look for combined balance column first (most common case)
    for col in df.columns:
        if col == column_mapping["account_name_col"] or col == column_mapping["account_number_col"]:
            continue
            
        col_lower = str(col).lower()
        
        # Check for balance patterns in column name
        if any(pattern.lower() in col_lower for pattern in balance_patterns if pattern):
            column_mapping["balance_type"] = "combined"
            column_mapping["balance_cols"] = {"balance": col}
            break
        
        # Check column content if name doesn't match
        if is_likely_balance_column(df[col], col):
            column_mapping["balance_type"] = "combined"
            column_mapping["balance_cols"] = {"balance": col}
            break
    
    def is_likely_debit_credit_pair(debit_col: str, credit_col: str, df: pd.DataFrame) -> bool:
        """Check if two columns form a valid debit/credit pair"""
        # Both columns should be numeric
        if not (is_numeric_column(df[debit_col]) and is_numeric_column(df[credit_col])):
            return False
            
        # Sample data to check patterns
        sample_size = min(100, len(df))
        sample = df[[debit_col, credit_col]].sample(sample_size)
        
        # Check if values are mutually exclusive (when one has value, other is zero/empty)
        debit_values = sample[debit_col].fillna(0).astype(float)
        credit_values = sample[credit_col].fillna(0).astype(float)
        
        # Count how many rows have values in both columns (should be rare in a proper debit/credit pair)
        both_non_zero = ((debit_values > 0) & (credit_values > 0)).sum()
        
        # If more than 10% of rows have values in both columns, they're probably not a debit/credit pair
        return (both_non_zero / sample_size) < 0.1
    
    # If no combined balance found, look for separate debit/credit columns
    if not column_mapping["balance_cols"].get("balance"):
        debit_patterns = [
            ("debit", 1.0), ("dr", 1.0), ("debit amount", 1.0), ("debits", 1.0),
            ("expense", 0.8), ("payment", 0.7), ("withdrawal", 0.7)  # Lower confidence patterns
        ]
        credit_patterns = [
            ("credit", 1.0), ("cr", 1.0), ("credit amount", 1.0), ("credits", 1.0),
            ("deposit", 0.8), ("income", 0.7), ("revenue", 0.7)  # Lower confidence patterns
        ]
        
        # Find all potential debit and credit columns with their confidence scores
        potential_debits = []
        for pattern, confidence in debit_patterns:
            matches = [(col, confidence) for col in columns 
                     if pattern.lower() in str(col).lower() 
                     and col != column_mapping["account_name_col"]]
            potential_debits.extend(matches)
        
        potential_credits = []
        for pattern, confidence in credit_patterns:
            matches = [(col, confidence) for col in columns 
                     if pattern.lower() in str(col).lower() 
                     and col != column_mapping["account_name_col"]]
            potential_credits.extend(matches)
        
        # Sort by confidence (highest first)
        potential_debits.sort(key=lambda x: x[1], reverse=True)
        potential_credits.sort(key=lambda x: x[1], reverse=True)
        
        # Try to find the best matching pair
        best_score = 0
        best_pair = (None, None)
        
        for debit_col, debit_conf in potential_debits[:5]:  # Limit to top 5 candidates
            for credit_col, credit_conf in potential_credits[:5]:
                if debit_col == credit_col:
                    continue  # Skip if same column
                    
                # Calculate combined confidence
                combined_conf = (debit_conf + credit_conf) / 2
                
                # Check if they form a valid debit/credit pair
                if is_likely_debit_credit_pair(debit_col, credit_col, df):
                    combined_conf *= 1.5  # Boost confidence for valid pairs
                
                if combined_conf > best_score:
                    best_score = combined_conf
                    best_pair = (debit_col, credit_col)
        
        # If we found a good enough pair, use it
        if best_score >= 1.0:  # At least one high-confidence match
            column_mapping["balance_type"] = "separate"
            column_mapping["balance_cols"] = {
                "debit": best_pair[0],
                "credit": best_pair[1]
            }
    
    # If we still haven't found a balance column, try to find any numeric column that could be a balance
    if not column_mapping["balance_cols"] or not column_mapping["balance_cols"].get("balance"):
        # Get all numeric columns that aren't the account name or number
        potential_cols = [col for col in df.columns 
                         if col != column_mapping["account_name_col"] 
                         and (column_mapping["account_number_col"] is None or col != column_mapping["account_number_col"])
                         and is_numeric_column(df[col])]
        
        if potential_cols:
            # Try to find a column with 'balance' in the name first
            for col in potential_cols:
                if 'balance' in str(col).lower():
                    column_mapping["balance_type"] = "combined"
                    column_mapping["balance_cols"] = {"balance": col}
                    break
            
            # If no 'balance' in name, try other balance-related patterns
            if not column_mapping["balance_cols"]:
                for pattern in balance_patterns[1:]:  # Skip 'sum of balance' since we already checked it
                    for col in potential_cols:
                        if pattern in str(col).lower():
                            column_mapping["balance_type"] = "combined"
                            column_mapping["balance_cols"] = {"balance": col}
                            break
                    if column_mapping["balance_cols"]:
                        break
            
            # If still no match, use the first numeric column that's not the account name or number
            if not column_mapping["balance_cols"] and potential_cols:
                column_mapping["balance_type"] = "combined"
                column_mapping["balance_cols"] = {"balance": potential_cols[0]}
        
        # If we have exactly two numeric columns and haven't found a match, assume they're debit/credit
        if len(potential_cols) == 2 and not column_mapping["balance_cols"].get("balance"):
            column_mapping["balance_type"] = "separate"
            column_mapping["balance_cols"] = {
                "debit": potential_cols[0],
                "credit": potential_cols[1]
            }
    
    return column_mapping

def classify_accounts(
    df: pd.DataFrame, 
    account_name_col: str, 
    account_number_col: Optional[str] = None,
    balance_cols: Dict[str, str] = None,
    batch_size: int = 25,
    confidence_threshold: int = 90
) -> pd.DataFrame:
    """
    Classify accounts using fuzzy search by default, and fall back to AI when confidence is low.
    Only uses AI when fuzzy match confidence is below the specified threshold.
    """
    # Make a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Add mapping columns if they don't exist
    if 'Global_Mapping' not in result_df.columns:
        result_df['Global_Mapping'] = ''
    if 'Specific_Mapping' not in result_df.columns:
        result_df['Specific_Mapping'] = ''
    if 'Confidence_Score' not in result_df.columns:
        result_df['Confidence_Score'] = 0.0
    if 'Done_By' not in result_df.columns:
        result_df['Done_By'] = ''
    
    # Calculate balance if needed for classification
    if balance_cols:
        if 'balance' in balance_cols and balance_cols['balance']:
            # Use the existing balance column
            balance_col = balance_cols['balance']
        elif 'debit' in balance_cols and 'credit' in balance_cols and balance_cols['debit'] and balance_cols['credit']:
            # Calculate balance from debit and credit
            result_df['Calculated_Balance'] = result_df.apply(
                lambda row: (row[balance_cols['debit']] if pd.notna(row[balance_cols['debit']]) else 0) - \
                           (row[balance_cols['credit']] if pd.notna(row[balance_cols['credit']]) else 0),
                axis=1
            )
            balance_col = 'Calculated_Balance'
        else:
            balance_col = None
    else:
        balance_col = None
    
    # Process accounts in batches
    total_rows = len(result_df)
    ai_candidates = []  # Store accounts that need AI classification
    
    # First pass: Try fuzzy matching for all accounts
    for idx, row in result_df.iterrows():
        account_name = str(row[account_name_col]).lower()
        
        # Get fuzzy matches with confidence scores
        global_match, global_confidence = fuzzy_match_global_category_with_confidence(account_name)
        
        if global_confidence >= confidence_threshold:
            # Use fuzzy match if confidence is high enough
            specific_match, specific_confidence = fuzzy_match_specific_category_with_confidence(
                account_name, global_match
            )
            
            result_df.at[idx, 'Global_Mapping'] = global_match
            result_df.at[idx, 'Specific_Mapping'] = specific_match
            result_df.at[idx, 'Confidence_Score'] = min(global_confidence, specific_confidence)
            result_df.at[idx, 'Done_By'] = 'fuzzy logic'
        else:
            # Store account for AI classification
            ai_candidates.append((idx, row))
    
    # Second pass: Process accounts that need AI classification in batches
    for i in range(0, len(ai_candidates), batch_size):
        batch = ai_candidates[i:i + batch_size]
        batch_indices = [item[0] for item in batch]
        batch_rows = [item[1] for item in batch]
        
        try:
            # Create a dataframe for the batch
            batch_df = pd.DataFrame(batch_rows)
            
            # Get AI classifications for the batch
            mapped_batch = ai_classify_accounts_batch(
                batch_df, 
                account_name_col, 
                account_number_col,
                balance_col
            )
            
            # Update the results with AI classifications
            # Ensure mapped_batch and batch_indices have the same length
            if len(mapped_batch) == len(batch_indices):
                for i in range(len(mapped_batch)):
                    row = mapped_batch.iloc[i]
                    original_df_index = batch_indices[i] # Use positional index i
                    
                    result_df.at[original_df_index, 'Global_Mapping'] = row.get('Global_Mapping', '')
                    result_df.at[original_df_index, 'Specific_Mapping'] = row.get('Specific_Mapping', '')
                    result_df.at[original_df_index, 'Confidence_Score'] = row.get('Confidence_Score', 100.0)
                    # Propagate 'Done_By' status from mapped_batch, default to 'AI' if not set by ai_classify_accounts_batch
                    result_df.at[original_df_index, 'Done_By'] = row.get('Done_By') if pd.notna(row.get('Done_By')) and row.get('Done_By') else 'AI'
            else:
                print(f"CRITICAL ERROR: Mismatch in lengths for AI batch processing!")
                print(f"len(mapped_batch) = {len(mapped_batch)}, len(batch_indices) = {len(batch_indices)}")
                # Fallback for this batch: mark as fuzzy logic due to batch error if not already AI
                for original_df_idx_fallback in batch_indices:
                    current_status = result_df.at[original_df_idx_fallback, 'Done_By']
                    if pd.isna(current_status) or current_status != 'AI': # Only override if not already AI processed successfully
                        result_df.at[original_df_idx_fallback, 'Done_By'] = 'fuzzy logic (batch error)'
                        result_df.at[original_df_idx_fallback, 'Confidence_Score'] = 0.0
        except Exception as e:
            print(f"Error during AI batch classification: {e}")
            # Fallback for the entire batch in case of unexpected error
            for original_df_idx_fallback in batch_indices:
                current_status = result_df.at[original_df_idx_fallback, 'Done_By']
                if pd.isna(current_status) or current_status != 'AI': # Only override if not already AI processed successfully
                    result_df.at[original_df_idx_fallback, 'Done_By'] = 'fuzzy logic (batch error)'
                    result_df.at[original_df_idx_fallback, 'Confidence_Score'] = 0.0

    # Standardize column names before returning
    # Standardize the account name column
    if account_name_col in result_df.columns and account_name_col != 'Original_Account_Name':
        result_df.rename(columns={account_name_col: 'Original_Account_Name'}, inplace=True)
        print(f"[classify_accounts] Renamed account column '{account_name_col}' to 'Original_Account_Name'")
    elif 'Original_Account_Name' not in result_df.columns and account_name_col not in result_df.columns:
        print(f"[classify_accounts] WARNING: Original account column '{account_name_col}' not found for renaming and 'Original_Account_Name' does not exist.")

    # Standardize the balance column
    # The 'balance_col' variable holds the name of the column that should become 'Balance'.
    # This could be the original combined balance column, or 'Calculated_Balance'.
    if balance_col and balance_col in result_df.columns and balance_col != 'Balance':
        result_df.rename(columns={balance_col: 'Balance'}, inplace=True)
        print(f"[classify_accounts] Renamed balance column '{balance_col}' to 'Balance'")
    elif balance_col and balance_col == 'Balance' and 'Balance' in result_df.columns:
        print(f"[classify_accounts] Balance column is already named 'Balance'. No rename needed.")
    elif balance_col and balance_col not in result_df.columns:
        print(f"[classify_accounts] WARNING: Identified balance column '{balance_col}' not found in DataFrame for renaming.")
    elif not balance_col and 'Balance' not in result_df.columns:
        print(f"[classify_accounts] WARNING: No balance column was identified or created to be standardized as 'Balance'.")


    return result_df

def ai_classify_accounts_batch(
    batch_df: pd.DataFrame, 
    account_name_col: str, 
    account_number_col: Optional[str] = None,
    balance_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Use AI to classify accounts into global and specific mapping categories.
    Processes each account individually with separate API calls.
    """
    # Check if client is initialized (API key is set)
    if client is None:
        print("Skipping AI classification: No OpenAI API key provided")
        return _mark_as_fuzzy(batch_df)
    
    # Create a copy of the dataframe to modify
    result_df = batch_df.copy()
    
    # Initialize columns if they don't exist
    if 'Global_Mapping' not in result_df.columns:
        result_df['Global_Mapping'] = ''
    if 'Specific_Mapping' not in result_df.columns:
        result_df['Specific_Mapping'] = ''
    if 'Done_By' not in result_df.columns:
        result_df['Done_By'] = ''
    if 'Confidence_Score' not in result_df.columns:
        result_df['Confidence_Score'] = 0.0
    
    for idx, row in result_df.iterrows():
        # Skip rows with empty account names
        if not row[account_name_col] or not str(row[account_name_col]).strip():
            continue
            
        account_name = str(row[account_name_col]).strip()
        
        # Skip common summary rows
        if account_name.lower() in ['grand total', 'row labels', 'total', 'sum', '']:
            continue
            
        try:
            # Prepare the account info for the prompt
            account_info = {"name": account_name}
            
            if account_number_col and account_number_col in row and pd.notna(row[account_number_col]):
                account_info["number"] = str(row[account_number_col]).strip()
                
            if balance_col and balance_col in row and pd.notna(row[balance_col]):
                account_info["balance"] = str(row[balance_col]).strip()
            
            # Create the prompt for a single account
            prompt = f"""Classify this account into appropriate global and specific mapping categories:

Account: {json.dumps(account_info["name"])}

Global mapping categories (choose exactly one):
- Asset
- Liability
- Equity
- Income
- Expense

Specific mapping categories should be more detailed, such as:
- For Assets: Cash and Cash Equivalents, Accounts Receivable, Inventory, Fixed Assets, etc.
- For Liabilities: Accounts Payable, Loans Payable, Accrued Expenses, etc.
- For Equity: Common Stock, Retained Earnings, etc.
- For Income: Sales Revenue, Interest Income, etc.
- For Expenses: Cost of Goods Sold, Salary Expense, Rent Expense, etc.

Return your analysis as a JSON object with these exact keys:
{{
  "name": "{json.dumps(account_info["name"])}",
  "global_mapping": "the global category",
  "specific_mapping": "the specific category"
}}

Be concise and only return the JSON object."""

            # Log the API call
            print(f"\n=== OpenAI API Call ===")
            print(f"Model: {AI_MODEL}")
            print(f"Account: {account_name}")
            print("Prompt:")
            print("-" * 40)
            print(prompt)
            print("-" * 40)
            
            try:
                # Make the API call for this single account
                start_time = time.time()
                response = client.chat.completions.create(
                    model=AI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an AI assistant specialized in accounting classification."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=200,
                    timeout=10  # 10 second timeout per account
                )
                end_time = time.time()
                
                # Log the response
                print(f"\n=== OpenAI API Response ===")
                print(f"Status: Success")
                print(f"Response Time: {(end_time - start_time):.2f}s")
                print(f"Model: {response.model}")
                print(f"Usage: {response.usage}")
                
                # Extract the response content
                if not response.choices or not response.choices[0].message.content:
                    print(f"No content in response for account: {account_name}")
                    continue
                    
                ai_response = response.choices[0].message.content
                print("Response Content:")
                print("-" * 40)
                print(ai_response)
                print("-" * 40)
                
            except Exception as e:
                print(f"\n=== OpenAI API Error ===")
                print(f"Error: {str(e)}")
                print(f"Account: {account_name}")
                import traceback
                traceback.print_exc()
                continue
            
            # Try to parse the JSON response
            try:
                # Try to find JSON object in the response
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = ai_response[json_start:json_end]
                    classification = json.loads(json_str)
                    
                    # Validate the classification
                    if not isinstance(classification, dict):
                        raise ValueError("Classification is not a JSON object")
                        
                    # Update the dataframe with the classification
                    result_df.at[idx, 'Global_Mapping'] = classification.get('global_mapping', '')
                    result_df.at[idx, 'Specific_Mapping'] = classification.get('specific_mapping', '')
                    result_df.at[idx, 'Done_By'] = 'AI'
                    result_df.at[idx, 'Confidence_Score'] = 100.0
                    
                    print(f"Classified '{account_name}' as {classification.get('global_mapping', '?')} / {classification.get('specific_mapping', '?')}")
                else:
                    print(f"Could not find JSON object in response for account: {account_name}")
                    
            except json.JSONDecodeError as je:
                print(f"Error decoding JSON for account '{account_name}': {str(je)}")
                print(f"Response was: {ai_response}")
                
        except Exception as e:
            print(f"Error processing account '{account_name}': {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Mark any unprocessed rows with fuzzy logic
    unprocessed = result_df[result_df['Done_By'] != 'AI']
    if not unprocessed.empty:
        print(f"Marking {len(unprocessed)} unprocessed rows as fuzzy logic")
        for idx in unprocessed.index:
            result_df.at[idx, 'Done_By'] = 'fuzzy logic'
    
    return result_df

def fuzzy_match_global_category_with_confidence(account_name: str) -> tuple[str, float]:
    """
    Use fuzzy matching to find the best global mapping category and its confidence score
    Returns a tuple of (best_category, confidence_score)
    """
    # Handle empty or whitespace-only account names
    if not account_name or not str(account_name).strip():
        return "Unknown", 0.0
        
    best_category = "Unknown"
    best_score = 0.0
    account_name = str(account_name).lower().strip()

    # Common word replacements and normalizations
    replacements = {
        'acct': 'account',
        'accts': 'accounts',
        'recv': 'receivable',
        'pay': 'payable',
        'exp': 'expense',
        'misc': 'miscellaneous',
        'eqpt': 'equipment',
        'int': 'interest',
        'admin': 'administrative',
        'maint': 'maintenance',
        'dept': 'department',
        'div': 'dividend',
        'inv': 'inventory',
        'mgr': 'manager',
        'mktg': 'marketing',
        'prod': 'product',
        'svc': 'service',
        'trd': 'trade',
        'util': 'utility',
        ' & ': ' and ',
        ' + ': ' and ',
        ' / ': ' ',
        '-': ' ',
        '_': ' '
    }
    
    # Clean and normalize the account name
    cleaned_name = account_name
    for old, new in replacements.items():
        cleaned_name = cleaned_name.replace(old, new)
    
    # Remove common prefixes/suffixes and clean up
    for prefix in ['total ', 'ending ', 'beginning ', 'closing ', 'opening ']:
        if cleaned_name.startswith(prefix):
            cleaned_name = cleaned_name[len(prefix):].strip()
    
    # Remove common suffixes
    for suffix in [' account', ' accounts', ' a/c', ' a c', ' acct', ' accts']:
        if cleaned_name.endswith(suffix):
            cleaned_name = cleaned_name[:-len(suffix)].strip()
    
    # If the cleaned name is empty after all replacements, use the original
    if not cleaned_name.strip():
        cleaned_name = account_name
    
    try:
        # First try exact matches for common patterns
        for category, patterns in GLOBAL_MAPPING_PATTERNS.items():
            for pattern in patterns:
                if pattern and (pattern in cleaned_name or pattern in account_name):
                    return category, 100.0  # Exact match
        
        # If no exact match, try fuzzy matching with cleaned name
        for category, patterns in GLOBAL_MAPPING_PATTERNS.items():
            for pattern in patterns:
                if not pattern.strip():  # Skip empty patterns
                    continue
                
                # Try both partial ratio and token set ratio for better matching
                partial_score = process.fuzz.partial_ratio(cleaned_name, pattern)
                token_score = process.fuzz.token_set_ratio(cleaned_name, pattern)
                
                # Use the higher of the two scores with some weighting
                score = max(partial_score, token_score * 0.9)
                
                # Boost score for certain patterns
                if pattern in ['receivable', 'payable', 'inventory'] and pattern in cleaned_name:
                    score = min(100, score * 1.1)
                
                if score > best_score:
                    best_score = score
                    best_category = category
        
        # Special handling for common patterns
        if best_score < 80:  # Only apply special handling if no good match found
            lower_name = cleaned_name.lower()
            
            # Service-related expenses
            if any(term in lower_name for term in ['service', 'support', 'maintenance', 'repair']):
                if any(term in lower_name for term in ['cost', 'cogs', 'expense']):
                    return 'Expense', 95.0
                return 'Expense', 90.0
                
            # Common expense patterns
            if any(term in lower_name for term in ['fee', 'charge', 'expense', 'cost', 'payment']):
                return 'Expense', 90.0
                
            # Common asset patterns
            if any(term in lower_name for term in ['cash', 'bank', 'account', 'receivable', 'inventory', 'asset']):
                return 'Asset', 90.0
                
            # Common liability patterns
            if any(term in lower_name for term in ['payable', 'debt', 'loan', 'credit']):
                return 'Liability', 90.0
        
        # If we have a decent match, return it, otherwise return Unknown
        if best_score >= 70:  # Lowered threshold from 80 to 70
            return best_category, best_score
            
        return "Unknown", 0.0
        
    except Exception as e:
        print(f"Error in fuzzy_match_global_category_with_confidence for '{account_name}': {str(e)}")
        return "Unknown", 0.0

def fuzzy_match_specific_category_with_confidence(account_name: str, global_category: str) -> tuple[str, float]:
    """
    Use fuzzy matching to find the best specific mapping category within a global category and its confidence score
    Returns a tuple of (best_category, confidence_score)
    """
    if not account_name or not str(account_name).strip() or global_category not in SPECIFIC_MAPPING_PATTERNS:
        return f"Other {global_category}", 0.0
    
    # Clean and normalize the account name
    account_name = str(account_name).lower().strip()
    specific_patterns = SPECIFIC_MAPPING_PATTERNS[global_category]
    best_category = f"Other {global_category}"
    best_score = 0.0
    
    # Common word replacements and normalizations (same as in global matching)
    replacements = {
        'acct': 'account', 'accts': 'accounts', 'recv': 'receivable', 'pay': 'payable',
        'exp': 'expense', 'misc': 'miscellaneous', 'eqpt': 'equipment', 'int': 'interest',
        'admin': 'administrative', 'maint': 'maintenance', 'dept': 'department',
        'div': 'dividend', 'inv': 'inventory', 'mgr': 'manager', 'mktg': 'marketing',
        'prod': 'product', 'svc': 'service', 'trd': 'trade', 'util': 'utility',
        ' & ': ' and ', ' + ': ' and ', ' / ': ' ', '-': ' ', '_': ' '
    }
    
    cleaned_name = account_name
    for old, new in replacements.items():
        cleaned_name = cleaned_name.replace(old, new)
    
    # Remove common prefixes/suffixes
    for prefix in ['total ', 'ending ', 'beginning ', 'closing ', 'opening ']:
        if cleaned_name.startswith(prefix):
            cleaned_name = cleaned_name[len(prefix):].strip()
    
    for suffix in [' account', ' accounts', ' a/c', ' a c', ' acct', ' accts']:
        if cleaned_name.endswith(suffix):
            cleaned_name = cleaned_name[:-len(suffix)].strip()
    
    if not cleaned_name.strip():
        cleaned_name = account_name
    
    # Special case for service-related expenses (moved to top for priority)
    if global_category == 'Expense' and ('service' in cleaned_name or 'service' in account_name):
        if any(term in cleaned_name or term in account_name for term in ['cost', 'cogs', 'sold']):
            return 'Cost of Services', 95.0
        return 'Professional Services', 90.0
    
    # First try exact matches against cleaned name
    for category, patterns in specific_patterns.items():
        for pattern in patterns:
            pattern_lower = pattern.lower()
            if pattern_lower in cleaned_name or pattern_lower in account_name:
                return category, 100.0  # Exact match
    
    # If no exact match, try fuzzy matching with multiple strategies
    for category, patterns in specific_patterns.items():
        for pattern in patterns:
            if not pattern.strip():
                continue
                
            # Try multiple matching strategies
            partial_score = process.fuzz.partial_ratio(cleaned_name, pattern.lower())
            token_score = process.fuzz.token_set_ratio(cleaned_name, pattern.lower())
            ratio_score = process.fuzz.ratio(cleaned_name, pattern.lower())
            
            # Use weighted combination of scores
            score = max(partial_score, token_score * 0.9, ratio_score * 0.8)
            
            # Boost score for certain patterns
            if 'receivable' in pattern and 'receivable' in cleaned_name:
                score = min(100, score * 1.1)
            elif 'payable' in pattern and 'payable' in cleaned_name:
                score = min(100, score * 1.1)
            
            if score > best_score:
                best_score = score
                best_category = category
    
    # Apply category-specific logic for better matching
    if global_category == 'Expense':
        lower_name = cleaned_name.lower()
        
        # Salary and wages
        if any(term in lower_name for term in ['salary', 'wage', 'payroll', 'compensation', 'employee', 'staff']):
            return 'Salary and Wages', 95.0
            
        # Rent and occupancy
        if any(term in lower_name for term in ['rent', 'lease', 'occupancy', 'facility']):
            return 'Rent and Occupancy', 95.0
            
        # Professional services
        if any(term in lower_name for term in ['legal', 'accounting', 'audit', 'consulting', 'professional', 'advisory']):
            return 'Professional Services', 95.0
            
        # Marketing and advertising
        if any(term in lower_name for term in ['marketing', 'advertising', 'promotion', 'branding', 'pr', 'public relation']):
            return 'Marketing and Advertising', 95.0
            
        # Technology and software
        if any(term in lower_name for term in ['software', 'technology', 'it ', 'hosting', 'cloud', 'saas', 'subscription']):
            return 'Technology and Software', 95.0
    
    # If we have a decent match, return it, otherwise return the default
    if best_score >= 70:  # Lowered threshold from 80 to 70
        return best_category, best_score
    
    # If we have a decent match in the global category, use that as a fallback
    if global_category == 'Expense':
        return 'General and Administrative', 75.0
    elif global_category == 'Asset':
        return 'Other Current Assets', 75.0
    elif global_category == 'Liability':
        return 'Other Current Liabilities', 75.0
    
    return best_category, best_score

def _mark_as_fuzzy(df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to mark accounts for fuzzy matching when AI classification fails"""
    result_df = df.copy()
    for idx, row in result_df.iterrows():
        if pd.isna(row.get('Done_By')) or row.get('Done_By') != 'AI':
            result_df.at[idx, 'Done_By'] = 'fuzzy logic'
            result_df.at[idx, 'Confidence_Score'] = 0.0
    return result_df

# Keep the old functions for backward compatibility
def fuzzy_match_global_category(account_name: str) -> str:
    """Legacy function that returns only the category name"""
    return fuzzy_match_global_category_with_confidence(account_name)[0]

def fuzzy_match_specific_category(account_name: str, global_category: str) -> str:
    """Legacy function that returns only the category name"""
    return fuzzy_match_specific_category_with_confidence(account_name, global_category)[0]
