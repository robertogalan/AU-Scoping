import os
import uuid
from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from utils.excel_processor import process_excel_file, save_mapped_excel
from utils.ai_mapper import identify_columns, classify_accounts
from utils.scoping import calculate_materiality, calculate_performance_materiality, scope_accounts, save_scoped_excel

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_mapping_app')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching of static files

# Add cache control headers to all responses
@app.after_request
def add_header(response):
    """Add headers to prevent caching"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/scoping/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@app.route('/scoping/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Store file info in session
        session['uploaded_file'] = {
            'original_name': original_filename,
            'path': filepath,
            'extension': file_extension
        }
        
        # Process the file for preview
        try:
            df = process_excel_file(filepath)
            preview_data = df.head(10).to_dict('records')
            columns = df.columns.tolist()
            
            # Store columns and dataframe in session (for later use)
            session['file_columns'] = columns
            
            # First try AI-based column identification
            try:
                column_mapping = identify_columns(df)
                ai_confidence = 1.0  # High confidence for AI-based selection
            except Exception as ai_error:
                print(f"AI column identification failed: {str(ai_error)}")
                # Fall back to basic identification if AI fails
                column_mapping = {
                    'account_name_col': columns[0] if columns else None,
                    'account_number_col': columns[1] if len(columns) > 1 else None,
                    'balance_type': 'combined',
                    'balance_cols': {'balance': columns[-1]} if columns else {}
                }
                ai_confidence = 0.0  # Low confidence for fallback
            
            # Store the AI's confidence level
            session['ai_confidence'] = ai_confidence
            
            return render_template(
                'preview.html', 
                preview_data=preview_data,
                columns=columns,
                column_mapping=column_mapping,
                ai_confidence=ai_confidence
            )
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    
    flash('File type not allowed')
    return redirect(url_for('index'))

@app.route('/process', methods=['POST'])
@app.route('/scoping/process', methods=['POST'])
def process_mapping():
    if 'uploaded_file' not in session:
        flash('No file uploaded')
        return redirect(url_for('index'))
    
    # Get column mapping from form
    account_name_col = request.form.get('account_name_col')
    account_number_col = request.form.get('account_number_col')
    debit_col = None  # Initialize to None
    credit_col = None # Initialize to None
    
    # Handle balance columns (could be separate debit/credit or combined)
    balance_type = request.form.get('balance_type')
    if balance_type == 'separate':
        debit_col = request.form.get('debit_col')
        credit_col = request.form.get('credit_col')
        balance_cols = {'debit': debit_col, 'credit': credit_col}
    else:  # combined
        balance_col = request.form.get('balance_col')
        balance_cols = {'balance': balance_col}
    
    specific_mapping_col = request.form.get('specific_mapping_column')
    global_mapping_col = request.form.get('global_mapping_column')

    app.logger.info(f"Values from /process form: account_name_col='{account_name_col}', account_number_col='{account_number_col}', balance_type='{balance_type}', debit_col='{debit_col}', credit_col='{credit_col}', specific_mapping_col='{specific_mapping_col}', global_mapping_col='{global_mapping_col}'")

    if not all([account_name_col, balance_cols['balance'] if balance_type == 'combined' else (debit_col and credit_col)]):
        flash('Please select all required columns')
        return redirect(url_for('index'))
    
    # Store mapping in session
    session['column_mapping'] = {
        'account_name': account_name_col,
        'account_number': account_number_col,
        'balance_type': balance_type,
        'balance_cols': balance_cols,
        'specific_mapping_col': specific_mapping_col,
        'global_mapping_col': global_mapping_col
    }
    
    # Process the file with the confirmed mapping
    try:
        filepath = session['uploaded_file']['path']
        df = process_excel_file(filepath)
        
        # Perform account classification using AI
        mapped_df = classify_accounts(
            df, 
            account_name_col=account_name_col,
            account_number_col=account_number_col,
            balance_cols=balance_cols
        )
        
        # Save the mapped file
        output_filepath = os.path.join(
            app.config['UPLOAD_FOLDER'], 
            f"mapped_{session['uploaded_file']['original_name']}"
        )
        save_mapped_excel(mapped_df, output_filepath)

        # Set the standardized column names to session for materiality steps
        session['original_account_column_name'] = 'Original_Account_Name' # Standardized output name
        session['balance_column_name'] = 'Balance'                     # Standardized output name
        session['specific_mapping_column_name'] = 'Specific_Mapping'     # Standardized output name
        session['global_mapping_column_name'] = 'Global_Mapping'         # Standardized output name
        
        app.logger.info(f"SESSION values SET by process_mapping: original_account_column_name='{session.get('original_account_column_name')}', balance_column_name='{session.get('balance_column_name')}', specific_mapping_column_name='{session.get('specific_mapping_column_name')}', global_mapping_column_name='{session.get('global_mapping_column_name')}'")

        # Store output path in session
        session['output_file'] = {
            'path': output_filepath,
            'name': f"mapped_{session['uploaded_file']['original_name']}"
        }
        
        # Prepare mapping summary for display
        mapping_summary = {
            'global_counts': mapped_df['Global_Mapping'].value_counts().to_dict(),
            'specific_counts': mapped_df['Specific_Mapping'].value_counts().to_dict(),
            'total_accounts': len(mapped_df)
        }
        
        return render_template(
            'mapping.html',
            mapping_summary=mapping_summary,
            preview_data=mapped_df.to_dict('records'),  # Show all rows instead of just first 20
            output_filename=session['output_file']['name']
        )
        
    except Exception as e:
        flash(f'Error in mapping process: {str(e)}')
        app.logger.error(f"Error during mapping process: {str(e)}", exc_info=True) # Add more detailed logging
        return redirect(url_for('upload_file'))

@app.route('/download')
@app.route('/scoping/download')
def download_mapped_file():
    if 'output_file' not in session:
        flash('No mapped file available')
        return redirect(url_for('index'))
    
    return send_from_directory(
        os.path.dirname(session['output_file']['path']),
        os.path.basename(session['output_file']['path']),
        as_attachment=True,
        download_name=session['output_file']['name']
    )

@app.route('/set_materiality', methods=['GET', 'POST'])
@app.route('/scoping/set_materiality', methods=['GET', 'POST'])
def set_materiality():
    app.logger.debug("Accessed set_materiality route")
    app.logger.info(f"SESSION at start of /set_materiality: original_account_column_name='{session.get('original_account_column_name')}', balance_column_name='{session.get('balance_column_name')}', specific_mapping_column_name='{session.get('specific_mapping_column_name')}', global_mapping_column_name='{session.get('global_mapping_column_name')}'")
    if 'output_file' not in session or not session['output_file']:
        flash("No mapped file found in session. Please upload a mapped trial balance first or ensure the previous steps were completed.", "warning")
        return redirect(url_for('upload_for_materiality'))

    output_file_info = session.get('output_file')
    app.logger.debug(f"In set_materiality: session['output_file'] is {output_file_info} of type {type(output_file_info)}")

    # Robust check for output_file_info structure
    if not (isinstance(output_file_info, dict) and 'path' in output_file_info and 'name' in output_file_info):
        app.logger.warning(f"In set_materiality: session['output_file'] data is invalid or incomplete. Value: {output_file_info}")
        flash_msg = "Session data for mapped file is invalid or incomplete. Please re-upload or start over."
        if isinstance(output_file_info, str):
            flash_msg += f" (Attempted to use path: {output_file_info})"
        flash(flash_msg, "danger")
        session.pop('output_file', None) # Clear bad data
        return redirect(url_for('upload_for_materiality'))

    mapped_file_path = output_file_info['path']
    file_name_for_logs_and_display = output_file_info['name']
    app.logger.debug(f"Processing file for materiality parameters: {file_name_for_logs_and_display} from path {mapped_file_path}")

    numeric_columns = []
    specific_benchmark_names = ['Revenue', 'Net Income', 'Total Assets', 'Total Equity', 'Operating Income'] 

    try:
        app.logger.debug(f"Attempting to load mapped file for dropdowns: {mapped_file_path}")
        df = pd.read_excel(mapped_file_path, engine='openpyxl')
        app.logger.debug(f"File loaded. Type of df: {type(df)}")

        if not isinstance(df, pd.DataFrame):
            app.logger.error(f"Loaded object from {mapped_file_path} is not a DataFrame. Type: {type(df)}. Content (first 100 chars if string): {str(df)[:100] if isinstance(df, str) else 'Not a string'}")
            # Use file_name_for_logs_and_display in flash message
            flash(f"Uploaded file '{file_name_for_logs_and_display}' could not be parsed into a valid table structure. It might be corrupted or not a standard Excel table.", "danger")
            # Return to allow flash message to display, or redirect to an error page/upload page
            return render_template('materiality.html', step='set_parameters', numeric_columns=[], specific_benchmark_names=specific_benchmark_names, filename=file_name_for_logs_and_display, error_critical=True)

        if df.empty:
            app.logger.warning(f"Loaded DataFrame from {mapped_file_path} ('{file_name_for_logs_and_display}') is empty.")
            flash(f"The file '{file_name_for_logs_and_display}' was loaded but appears to be empty or could not be read correctly.", "warning")
            # Let template handle empty numeric_columns
        else:
            app.logger.debug(f"DataFrame columns from '{file_name_for_logs_and_display}': {df.columns.tolist()}")
            numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
            app.logger.debug(f"Numeric columns initially identified: {numeric_columns}")
            
            if 'Balance' in df.columns:
                if pd.api.types.is_numeric_dtype(df['Balance']):
                    if 'Balance' not in numeric_columns:
                        numeric_columns.append('Balance')
                        app.logger.debug(f"Added existing numeric 'Balance' to dropdown options.")
                else:
                    try:
                        # Try converting to numeric to check if it's possible
                        pd.to_numeric(df['Balance'])
                        if 'Balance' not in numeric_columns:
                            numeric_columns.append('Balance')
                            app.logger.debug(f"Added convertible 'Balance' to dropdown options.")
                    except ValueError:
                        app.logger.warning(f"'Balance' column in '{file_name_for_logs_and_display}' ({mapped_file_path}) is present but not numeric and could not be converted.")
            else:
                app.logger.warning(f"'Balance' column not found in '{file_name_for_logs_and_display}' ({mapped_file_path}).")
            
            # Populate specific_benchmark_names from 'Specific_Mapping' or 'Global_Mapping'
            current_specific_names = list(specific_benchmark_names) # Start with defaults
            if session.get('specific_mapping_column_name', 'Specific_Mapping') in df.columns:
                current_specific_names.extend(df[session.get('specific_mapping_column_name', 'Specific_Mapping')].astype(str).str.strip().dropna().unique().tolist())
            if session.get('global_mapping_column_name', 'Global_Mapping') in df.columns:
                current_specific_names.extend(df[session.get('global_mapping_column_name', 'Global_Mapping')].astype(str).str.strip().dropna().unique().tolist())
            specific_benchmark_names = sorted(list(set(name for name in current_specific_names if name and name.strip())))
            if not specific_benchmark_names or len(specific_benchmark_names) > 30:
                 specific_benchmark_names = ['Revenue', 'Net Income', 'Total Assets', 'Total Equity', 'Operating Income'] # Fallback

    except TypeError as e:
        # Use file_name_for_logs_and_display in log and flash messages
        if "string indices must be integers" in str(e):
            app.logger.error(f"Critical TypeError (string indices) processing Excel file '{file_name_for_logs_and_display}' from path {mapped_file_path}: {str(e)}", exc_info=True)
            flash(f"Error processing file structure for '{file_name_for_logs_and_display}': {str(e)}. The Excel file might be corrupted or have an unusual internal format. Try re-saving it or checking its integrity.", "danger")
        else:
            app.logger.error(f"Unexpected TypeError processing file '{file_name_for_logs_and_display}' from path {mapped_file_path}: {str(e)}", exc_info=True)
            flash(f"An unexpected type error occurred with file '{file_name_for_logs_and_display}': {str(e)}.", "danger")
    except ValueError as e: 
        app.logger.error(f"ValueError loading file '{file_name_for_logs_and_display}' from path {mapped_file_path}: {str(e)}", exc_info=True)
        flash(f"Error loading mapped file '{file_name_for_logs_and_display}': {str(e)}. Please ensure it's a valid Excel file and contains data.", "danger")
    except Exception as e:
        app.logger.error(f"General exception loading file '{file_name_for_logs_and_display}' from path {mapped_file_path}: {str(e)}", exc_info=True)
        flash(f"Error loading mapped file '{file_name_for_logs_and_display}': {str(e)}. Please ensure it's a valid Excel file.", "danger")

    materiality_results = session.get('materiality_results')
    app.logger.debug(f"In set_materiality, passing materiality_results to template: {materiality_results}")
    return render_template('materiality.html', 
                           step='set_parameters', 
                           numeric_columns=numeric_columns, 
                           specific_benchmark_names=specific_benchmark_names,
                           filename=file_name_for_logs_and_display,
                           materiality_results=materiality_results)

@app.route('/calculate_materiality', methods=['POST'])
@app.route('/scoping/calculate_materiality', methods=['POST'])
def calculate_materiality_route():
    app.logger.info(f"SESSION at start of /calculate_materiality: original_account_column_name='{session.get('original_account_column_name')}', balance_column_name='{session.get('balance_column_name')}', specific_mapping_column_name='{session.get('specific_mapping_column_name')}', global_mapping_column_name='{session.get('global_mapping_column_name')}'")
    """Calculate materiality and perform scoping"""
    if 'output_file' not in session:
        flash('You need to map accounts first', 'warning')
        return redirect(url_for('index'))
    
    try:
        # Get form parameters
        benchmark_type = request.form.get('benchmark_type')
        benchmark_column = request.form.get('benchmark_column')
        percentage = float(request.form.get('percentage'))
        pm_percentage = float(request.form.get('pm_percentage'))
        
        # Load the mapped data
        mapped_file_path = session['output_file']['path']
        app.logger.debug(f"Attempting to load mapped file for calculation: {mapped_file_path}")
        df = pd.read_excel(mapped_file_path, engine='openpyxl')
        app.logger.debug(f"File loaded for calculation. Type of df: {type(df)}")

        if not isinstance(df, pd.DataFrame):
            app.logger.error(f"Loaded object from {mapped_file_path} for calculation is not a DataFrame. Type: {type(df)}. Content (first 100 chars if string): {str(df)[:100] if isinstance(df, str) else 'Not a string'}")
            raise ValueError("Uploaded file for calculation could not be parsed into a valid table structure. It might be corrupted or not a standard Excel table.")
        if df.empty:
            app.logger.error(f"Loaded DataFrame from {mapped_file_path} for calculation is empty.")
            raise ValueError("Uploaded file for calculation is empty or could not be read properly.")

        balance_col_name_from_session = session.get('balance_column_name') 
        app.logger.debug(f"Retrieved balance_column_name from session: {balance_col_name_from_session}")

        if not balance_col_name_from_session or balance_col_name_from_session not in df.columns:
            error_msg = f"Balance column '{balance_col_name_from_session if balance_col_name_from_session else 'Not Set In Session'}' not found in DataFrame from {mapped_file_path}. Available columns: {df.columns.tolist()}"
            app.logger.error(error_msg)
            flash(f"Error: The required balance column (expected: '{balance_col_name_from_session if balance_col_name_from_session else 'Not Set'}') was not found in the mapped file or not set correctly in the session. Please ensure the file has the correct balance column and it was identified during the mapping or upload process.", "danger")
            return redirect(url_for('set_materiality'))
        
        if not pd.api.types.is_numeric_dtype(df[balance_col_name_from_session]):
            app.logger.info(f"Balance column '{balance_col_name_from_session}' is not numeric (type: {df[balance_col_name_from_session].dtype}). Attempting conversion.")
            try:
                df[balance_col_name_from_session] = pd.to_numeric(df[balance_col_name_from_session])
                app.logger.info(f"Successfully converted column '{balance_col_name_from_session}' to numeric.")
            except ValueError as ve:
                app.logger.error(f"ValueError converting balance column '{balance_col_name_from_session}' to numeric: {ve}", exc_info=True)
                flash(f"Error: The balance column '{balance_col_name_from_session}' contains non-numeric values that could not be converted. Please check the data in this column.", "danger")
                return redirect(url_for('set_materiality'))
        
        app.logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        app.logger.debug(f"Balance column ('{balance_col_name_from_session}') type: {df[balance_col_name_from_session].dtype}")
        # ... (rest of the logging and calculation logic from original block) ...
        if not df.empty:
            app.logger.debug(f"First 5 Balance values: {df[balance_col_name_from_session].head().tolist()}")
            app.logger.debug(f"Balance column sum: {df[balance_col_name_from_session].sum()}")
        if 'Global_Mapping' in df.columns:
            app.logger.debug(f"First 5 Global_Mapping values: {df['Global_Mapping'].head().tolist()}")
            app.logger.debug(f"Unique Global_Mapping values: {df['Global_Mapping'].unique().tolist()}")
        
        relevant_cols_for_sample = [session.get('original_account_column_name'), 
                                    session.get('account_number_column_name'), 
                                    balance_col_name_from_session, 
                                    'Global_Mapping', 
                                    'Specific_Mapping']
        cols_to_log = [col for col in relevant_cols_for_sample if col and col in df.columns]
        if cols_to_log:
            app.logger.debug(f"DataFrame sample for materiality (first 5 rows of relevant columns):\n{df[cols_to_log].head()}")
        else:
            app.logger.warning("Could not log DataFrame sample as no relevant columns were found.")

        original_account_col = session.get('original_account_column_name')
        app.logger.debug(f"Original account column from session for calculation: {original_account_col}")

        materiality_result = calculate_materiality(
            df,
            benchmark_column_input=benchmark_column,
            benchmark_type=benchmark_type,
            percentage=percentage,
            account_name_col=original_account_col,
            specific_mapping_col=session.get('specific_mapping_column_name', 'Specific_Mapping'),
            global_mapping_col=session.get('global_mapping_column_name', 'Global_Mapping'),
            balance_col=balance_col_name_from_session
        )

        if materiality_result is None or 'error' in materiality_result:
            error_message = materiality_result.get('error', 'Unknown error during calculation.') if isinstance(materiality_result, dict) else 'Unknown error during calculation.'
            calculation_steps = materiality_result.get('calculation_steps', []) if isinstance(materiality_result, dict) else []
            app.logger.error(f"Materiality calculation failed. Error: {error_message}. Steps: {calculation_steps}")
            flash(f"Materiality calculation failed: {error_message}", "danger")
            if calculation_steps:
                flash("Calculation attempt details: " + " -> ".join(calculation_steps), "info")
            return redirect(url_for('set_materiality'))
        
        overall_materiality = materiality_result.get('overall_materiality')
        benchmark_value = materiality_result.get('benchmark_value')
        calculation_steps = materiality_result.get('calculation_steps', [])

        if overall_materiality is None:
            flash("Overall materiality could not be determined from calculation results.", "danger")
            app.logger.error("Overall materiality is None after successful calculate_materiality call, but no error reported in materiality_result.")
            # Add current materiality_result to log for debugging
            app.logger.error(f"Problematic materiality_result: {materiality_result}")
            return redirect(url_for('set_materiality'))

        # --- Calculate Performance Materiality ---
        DEFAULT_PM_PERCENTAGE = 75  # e.g., 75%
        pm_calculation_result = calculate_performance_materiality(overall_materiality, DEFAULT_PM_PERCENTAGE)

        performance_materiality = None
        performance_materiality_percentage = DEFAULT_PM_PERCENTAGE # Default to attempted percentage

        if pm_calculation_result and 'error' not in pm_calculation_result:
            performance_materiality = pm_calculation_result.get('performance_materiality')
            performance_materiality_percentage = pm_calculation_result.get('performance_materiality_percentage', DEFAULT_PM_PERCENTAGE)
        elif pm_calculation_result and 'error' in pm_calculation_result:
            error_message_pm = pm_calculation_result.get('error', 'Unknown error during Performance Materiality calculation.')
            flash(f"Warning: Performance Materiality calculation failed: {error_message_pm}", "warning")
            app.logger.warning(f"Performance Materiality calculation failed: {error_message_pm}. Overall Materiality will still be used.")
        else: # pm_calculation_result is None or unexpected structure
            flash("Warning: Performance Materiality calculation returned an unexpected result.", "warning")
            app.logger.warning("Performance Materiality calculation returned None or unexpected structure. Overall Materiality will still be used.")

        # --- Calculate Clearly Trivial Threshold ---
        DEFAULT_CT_PERCENTAGE_OF_OM = 5  # e.g., 5%
        clearly_trivial_threshold = (overall_materiality * (DEFAULT_CT_PERCENTAGE_OF_OM / 100)) if overall_materiality is not None else None
        
        # --- Construct the full results dictionary ---
        # 'benchmark_column' and 'percentage' are from the form, available in this scope
        final_materiality_results = {
            'overall_materiality': overall_materiality,
            'performance_materiality': performance_materiality,
            'performance_materiality_percentage': performance_materiality_percentage,
            'benchmark_name': benchmark_column,  # User's benchmark input string
            'benchmark_value': benchmark_value,
            'percentage_used': float(percentage),  # User's percentage input for OM
            'clearly_trivial_threshold': clearly_trivial_threshold,
            'calculation_steps': calculation_steps
        }
        
        session['materiality_results'] = final_materiality_results
        app.logger.info(f"Overall Materiality: {final_materiality_results.get('overall_materiality')}")
        app.logger.info(f"Performance Materiality: {final_materiality_results.get('performance_materiality')} ({final_materiality_results.get('performance_materiality_percentage')}%)")
        app.logger.info(f"Clearly Trivial Threshold: {final_materiality_results.get('clearly_trivial_threshold')}")
        app.logger.debug(f"Full materiality results stored in session: {final_materiality_results}")

        return redirect(url_for('set_materiality'))

    except TypeError as e:
        if "string indices must be integers" in str(e):
            app.logger.error(f"Critical TypeError (string indices) in materiality calculation: {e}", exc_info=True)
            flash(f"Error in materiality calculation: {str(e)}. This often indicates an issue with the uploaded file's format not being correctly interpreted as a table.", "danger")
        else:
            app.logger.error(f"An unexpected TypeError occurred: {e}", exc_info=True)
            flash(f"An unexpected type error occurred during materiality calculation: {str(e)}", "danger")
        return redirect(url_for('set_materiality'))
    except ValueError as e:
        app.logger.error(f"ValueError in materiality calculation: {e}", exc_info=True)
        flash(f"Error in materiality calculation: {str(e)}. This could be due to data conversion issues or problems with the file structure.", "danger")
        return redirect(url_for('set_materiality'))
    except Exception as e:
        app.logger.error(f"Unexpected error in calculate_materiality_route: {e}", exc_info=True)
        flash(f"An unexpected error occurred: {str(e)}. Please check the application logs for more details.", "danger")
        return redirect(url_for('set_materiality')) # Ensure redirect happens immediately
            
        print(f"Materiality calculated successfully: {materiality:,.2f}")
        
        # Calculate performance materiality with error handling
        try:
            performance_materiality = calculate_performance_materiality(materiality, pm_percentage)
            print(f"Performance materiality calculated: {performance_materiality:,.2f}")
        except Exception as e:
            error_msg = f"Error calculating performance materiality: {str(e)}"
            print(f"ERROR: {error_msg}")
            flash(error_msg, 'danger')
            return redirect(url_for('set_materiality'))
        
        # Perform scoping
        scoped_df = scope_accounts(df, performance_materiality)
        
        # Store scoping information in session
        session['scoping'] = {
            'benchmark': {
                'type': benchmark_type,
                'column': benchmark_column,
                'value': benchmark_value,
                'percentage': percentage
            },
            'materiality': {
                'value': materiality,
                'pm_percentage': pm_percentage,
                'performance_materiality': performance_materiality
            }
        }
        
        # Save the scoped dataframe to a new file
        scoped_output_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 
            f"scoped_{session['uploaded_file']['original_name']}"
        )
        save_scoped_excel(scoped_df, scoped_output_path)
        
        # Store the scoped file path in session
        session['scoped_file'] = {
            'path': scoped_output_path,
            'name': f"scoped_{session['uploaded_file']['original_name']}"
        }
        
        # Calculate scope summary
        scope_summary = {
            'total': len(scoped_df),
            'in_scope': len(scoped_df[scoped_df['Final_Scope'] == 'In Scope']),
            'out_of_scope': len(scoped_df[scoped_df['Final_Scope'] == 'Out of Scope'])
        }
        
        # Prepare account data for template
        accounts_data = scoped_df.to_dict('records')
        
        # Rename columns for display if needed
        column_mapping = session['column_mapping']
        account_name_col = column_mapping['account_name']
        account_number_col = column_mapping['account_number']
        
        for account in accounts_data:
            account['Account_Name'] = account.get(account_name_col, '')
            account['Account_Number'] = account.get(account_number_col, '')
        
        return render_template(
            'materiality.html',
            step='review_scoping',
            benchmark_info=session['scoping']['benchmark'],
            materiality_info=session['scoping']['materiality'],
            scope_summary=scope_summary,
            accounts=accounts_data
        )
        
    except Exception as e:
        flash(f'Error in materiality calculation: {str(e)}', 'danger')
        return redirect(url_for('set_materiality'))

@app.route('/finalize_scoping', methods=['POST'])
@app.route('/scoping/finalize_scoping', methods=['POST'])
def finalize_scoping():
    """Finalize the scoping process"""
    if 'scoped_file' not in session:
        flash('You need to calculate materiality first', 'warning')
        return redirect(url_for('set_materiality'))
    
    try:
        # Load the scoped data for summary
        scoped_file_path = session['scoped_file']['path']
        df = pd.read_excel(scoped_file_path) if scoped_file_path.endswith(('xlsx', 'xls')) else pd.read_csv(scoped_file_path)
        
        # Calculate final scope summary
        in_scope_count = len(df[df['Final_Scope'] == 'In Scope'])
        total_count = len(df)
        
        scope_summary = {
            'total': total_count,
            'in_scope': in_scope_count,
            'out_of_scope': total_count - in_scope_count,
            'in_scope_percentage': (in_scope_count / total_count * 100) if total_count > 0 else 0
        }
        
        return render_template(
            'materiality.html',
            step='final',
            benchmark_info=session['scoping']['benchmark'],
            materiality_info=session['scoping']['materiality'],
            scope_summary=scope_summary,
            output_filename=session['scoped_file']['name']
        )
        
    except Exception as e:
        flash(f'Error finalizing scoping: {str(e)}', 'danger')
        return redirect(url_for('calculate_materiality'))

@app.route('/download_scoped')
@app.route('/scoping/download_scoped')
def download_scoped_file():
    """Download the scoped Excel file"""
    if 'scoped_file' not in session:
        flash('No scoped file available', 'warning')
        return redirect(url_for('index'))
    
    return send_from_directory(
        os.path.dirname(session['scoped_file']['path']),
        os.path.basename(session['scoped_file']['path']),
        as_attachment=True,
        download_name=session['scoped_file']['name']
    )

@app.route('/upload_for_materiality', methods=['POST'])
@app.route('/scoping/upload_for_materiality', methods=['POST'])
def upload_for_materiality():
    if 'mapped_file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('set_materiality', step='upload_mapped_file_prompt'))
    file = request.files['mapped_file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('set_materiality', step='upload_mapped_file_prompt'))
    if file and (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        original_filename = secure_filename(file.filename)
        unique_prefix = f"mapped_{uuid.uuid4().hex[:8]}_"
        filename = unique_prefix + original_filename
        
        direct_uploads_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'materiality_direct_uploads')
        os.makedirs(direct_uploads_dir, exist_ok=True)
        
        filepath = os.path.join(direct_uploads_dir, filename)
        file.save(filepath)
        
        session['output_file'] = {'name': original_filename, 'path': filepath}
        app.logger.info(f"Uploaded mapped file for materiality: {original_filename} to {filepath}")

        # Attempt to identify columns from this uploaded mapped file
        try:
            df = pd.read_excel(filepath)
            if not df.empty:
                # Simplified column identification for directly uploaded mapped files
                # Heuristic: first column is likely account name/description
                # Look for a column that sounds like 'balance' or 'amount'
                potential_account_cols = [col for col in df.columns if 'label' in col.lower() or 'name' in col.lower() or 'desc' in col.lower() or 'account' in col.lower()]
                session['original_account_column_name'] = potential_account_cols[0] if potential_account_cols else df.columns[0]
                
                potential_balance_cols = [col for col in df.columns if 'balance' in col.lower() or 'amount' in col.lower() or 'value' in col.lower()]
                if potential_balance_cols:
                    session['balance_column_name'] = potential_balance_cols[0]
                elif 'Sum of Balance' in df.columns: # Specific check for user's case
                    session['balance_column_name'] = 'Sum of Balance'
                else: # Fallback to a common name or a numeric column
                    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                    session['balance_column_name'] = numeric_cols[0] if numeric_cols else df.columns[1] if len(df.columns) > 1 else df.columns[0]

                session['specific_mapping_column_name'] = 'Specific_Mapping' # Assume standard name
                session['global_mapping_column_name'] = 'Global_Mapping'   # Assume standard name
                app.logger.info(f"SESSION after /upload_for_materiality column identification: original_account_column_name='{session.get('original_account_column_name')}', balance_column_name='{session.get('balance_column_name')}', specific_mapping_column_name='{session.get('specific_mapping_column_name')}', global_mapping_column_name='{session.get('global_mapping_column_name')}'")
                flash(f'Uploaded and processed columns from {original_filename} successfully.', 'success')
            else:
                flash(f'Uploaded {original_filename}, but it appears to be empty.', 'warning')
        except Exception as e:
            app.logger.error(f"Error processing directly uploaded mapped file '{original_filename}' for column identification: {str(e)}", exc_info=True)
            flash(f"Uploaded {original_filename}, but encountered an error trying to identify its columns: {str(e)}. Please ensure it's a valid mapped file.", 'danger')
            # Clear potentially problematic session vars if processing failed
            session.pop('original_account_column_name', None)
            session.pop('balance_column_name', None)

        flash(f'Uploaded {filename} successfully for materiality calculation.', 'success')
        return redirect(url_for('set_materiality'))
    else:
        flash('Invalid file type. Please upload Excel or CSV files.', 'danger')
        return redirect(url_for('set_materiality'))

if __name__ == '__main__':
    app.run(debug=True, port=5007)
