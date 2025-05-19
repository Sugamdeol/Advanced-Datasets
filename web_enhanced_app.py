import os
import csv
import json
import tempfile
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, send_file, session, jsonify
import google.generativeai as genai
from io import StringIO
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from web_utils import create_dataset_from_web_search, perform_duckduckgo_search

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Available models that can be used
AVAILABLE_MODELS = {
    'gemini-2.5-flash-preview-04-17': 'Gemini 2.5 Flash (Default)',
    'gemini-1.5-flash': 'Gemini 1.5 Flash',
    'gemini-1.5-pro': 'Gemini 1.5 Pro'
}

# Initialize default model name
DEFAULT_MODEL = 'gemini-2.5-flash-preview-04-17'

def configure_genai_with_api_key(api_key):
    """Configure the Gemini API with the provided key"""
    if not api_key:
        return False
    try:
        genai.configure(api_key=api_key)
        # Test the API key by creating a model
        test_model = genai.GenerativeModel(DEFAULT_MODEL)
        return True
    except Exception as e:
        print(f"Error configuring Gemini API: {str(e)}")
        return False

# Create directories for uploads and visualizations
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
VIS_FOLDER = os.path.join(STATIC_FOLDER, 'visualizations')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(VIS_FOLDER, exist_ok=True)

# In-memory database to store generated datasets
datasets_db = []
last_dataset_id = 0

def generate_dataset(prompt, num_rows, columns, model_name=DEFAULT_MODEL, schema=None, previous_data=None, use_web_search=False):
    """Generate dataset using Gemini API with optional web search"""
    global last_dataset_id
    
    try:
        if use_web_search:
            # Generate dataset using web search results
            df, search_results = create_dataset_from_web_search(prompt, columns, num_rows)
            if df is None:
                raise Exception(f"Web search failed: {search_results}")
                
            # Create a unique dataset ID
            last_dataset_id += 1
            dataset_id = last_dataset_id
            
            # Save to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"dataset_{dataset_id}_{timestamp}.csv"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            df.to_csv(file_path, index=False)
            
            # Store dataset in the database
            dataset_info = {
                'id': dataset_id,
                'name': f"Web Dataset: {prompt[:30]}...",
                'description': prompt,
                'columns': columns,
                'row_count': len(df),
                'file_path': file_path,
                'file_name': filename,
                'format': 'csv',
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_used': model_name,
                'web_search_used': True,
                'search_results': search_results
            }
            
            datasets_db.append(dataset_info)
            
            return df, dataset_info
        else:
            # Use the specified model
            current_model = genai.GenerativeModel(model_name)
            
            # Build column specifications with schema information if available
            column_specs = []
            for i, col_name in enumerate(columns):
                col_spec = col_name
                if schema and i < len(schema):
                    col_info = schema[i]
                    if 'type' in col_info:
                        col_spec += f" ({col_info['type']})"
                    if 'constraints' in col_info:
                        constraints = ', '.join([f"{k}={v}" for k, v in col_info['constraints'].items()])
                        if constraints:
                            col_spec += f" [{constraints}]"
                column_specs.append(col_spec)
            
            # Prepare the prompt for Gemini
            full_prompt = f"""
            Generate a high-quality dataset with {num_rows} rows and the following columns: {', '.join(column_specs)}.
            
            Format the response as CSV data without markdown formatting.
            Do not include any explanations, just return raw CSV data.
            
            Dataset description: {prompt}
            """
            
            # If continuing from previous data, include that context
            if previous_data:
                full_prompt += f"\n\nThis is a continuation. Here's what has been generated so far (summary):\n{previous_data}\nPlease continue from where this left off, maintaining consistency with previous data."
            
            # Generate content from Gemini
            response = current_model.generate_content(full_prompt)
            
            # Extract the CSV data from response
            csv_data = response.text.strip()
            
            # Clean up the response if needed (remove markdown code blocks if present)
            if csv_data.startswith("```") and csv_data.endswith("```"):
                csv_data = csv_data[3:-3].strip()
            if csv_data.startswith("```csv") and csv_data.endswith("```"):
                csv_data = csv_data[6:-3].strip()
                
            return csv_data, None
    except Exception as e:
        raise Exception(f"Error generating dataset: {str(e)}")

def process_csv_data(csv_data, columns):
    """Process CSV data into a pandas DataFrame"""
    try:
        # Parse CSV data
        csv_io = StringIO(csv_data)
        reader = csv.reader(csv_io)
        
        # Skip header if it exists and matches columns
        rows = list(reader)
        data = []
        
        # Check if first row matches column headers
        if rows and all(header.lower().strip() == col.lower().strip() for header, col in zip(rows[0], columns)):
            data = rows[1:]  # Skip header
        else:
            data = rows  # Use all rows
            
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # If DataFrame has more columns than expected, trim it
        if len(df.columns) > len(columns):
            df = df.iloc[:, :len(columns)]
            
        # If DataFrame has fewer columns than expected, add empty ones
        elif len(df.columns) < len(columns):
            for i in range(len(df.columns), len(columns)):
                df[i] = ""
                
        # Set column names
        df.columns = columns
        
        return df
    except Exception as e:
        raise Exception(f"Error processing CSV data: {str(e)}")



def generate_visualizations(df, dataset_id):
    """Generate visualizations for the dataset"""
    charts = []
    
    try:
        # Get column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Generate histograms for numeric columns
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            try:
                # Create a unique filename for this chart
                chart_id = f"hist_{col}_{dataset_id}"
                chart_path = os.path.join('static', 'visualizations', f"{chart_id}.png")
                
                # Create histogram using Seaborn
                plt.figure(figsize=(8, 5))
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f"Distribution of {col}")
                plt.tight_layout()
                plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), chart_path))
                plt.close()
                
                charts.append({
                    'id': chart_id,
                    'title': f"Distribution of {col}",
                    'type': 'histogram',
                    'path': chart_path
                })
            except Exception as e:
                print(f"Error generating histogram for {col}: {str(e)}")
        
        # Generate bar charts for categorical columns (top 10 values)
        for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
            try:
                # Only proceed if column has 20 or fewer unique values
                if df[col].nunique() <= 20:
                    # Create a unique filename for this chart
                    chart_id = f"bar_{col}_{dataset_id}"
                    chart_path = os.path.join('static', 'visualizations', f"{chart_id}.png")
                    
                    # Create bar chart
                    plt.figure(figsize=(10, 6))
                    value_counts = df[col].value_counts().nlargest(10)
                    sns.barplot(x=value_counts.index, y=value_counts.values)
                    plt.title(f"Top 10 values for {col}")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), chart_path))
                    plt.close()
                    
                    charts.append({
                        'id': chart_id,
                        'title': f"Top 10 values for {col}",
                        'type': 'bar',
                        'path': chart_path
                    })
            except Exception as e:
                print(f"Error generating bar chart for {col}: {str(e)}")
        
        # Generate scatter plot if we have at least 2 numeric columns
        if len(numeric_cols) >= 2:
            try:
                chart_id = f"scatter_{numeric_cols[0]}_{numeric_cols[1]}_{dataset_id}"
                chart_path = os.path.join('static', 'visualizations', f"{chart_id}.png")
                
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1])
                plt.title(f"Scatter plot: {numeric_cols[0]} vs {numeric_cols[1]}")
                plt.tight_layout()
                plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), chart_path))
                plt.close()
                
                charts.append({
                    'id': chart_id,
                    'title': f"Correlation: {numeric_cols[0]} vs {numeric_cols[1]}",
                    'type': 'scatter',
                    'path': chart_path
                })
            except Exception as e:
                print(f"Error generating scatter plot: {str(e)}")
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
        
    return charts

@app.route('/', methods=['GET', 'POST'])
def index():
    # Get theme preference from session or default to light
    theme = session.get('theme', 'light')
    
    if request.method == 'POST':
        # Get and store the API key
        api_key = request.form.get('api_key', '')
        if api_key:
            session['api_key'] = api_key
            # Configure Gemini API with the provided key
            if not configure_genai_with_api_key(api_key):
                flash("Invalid API key or error connecting to Gemini API. Please check your API key and try again.")
                return redirect(url_for('index'))
        elif 'api_key' not in session:
            flash("Please provide a valid Gemini API key.")
            return redirect(url_for('index'))
        else:
            # Use the API key from the session
            configure_genai_with_api_key(session['api_key'])
            
        prompt = request.form.get('prompt', '')
        columns_input = request.form.get('columns', '')
        num_rows = int(request.form.get('num_rows', 50))
        model_name = request.form.get('model', DEFAULT_MODEL)
        output_format = request.form.get('output_format', 'csv')
        use_web_search = 'use_web_search' in request.form
        
        # Schema definition (data types and constraints)
        schema_json = request.form.get('schema_json', '[]')
        try:
            schema = json.loads(schema_json)
        except json.JSONDecodeError:
            schema = []
        
        # Quality options
        missing_values_pct = float(request.form.get('missing_values_pct', 0))
        unique_columns = request.form.getlist('unique_columns')
        
        if not prompt or not columns_input:
            flash("Please provide both a prompt and columns")
            return redirect(url_for('index'))
            
        # Parse columns
        columns = [col.strip() for col in columns_input.split(',') if col.strip()]
        
        # Create schema if not provided
        if not schema:
            schema = [{'name': col} for col in columns]
        
        # Update schema with unique constraints if specified
        for col_schema in schema:
            if col_schema['name'] in unique_columns:
                if 'constraints' not in col_schema:
                    col_schema['constraints'] = {}
                col_schema['constraints']['unique'] = True
        
        try:
            # Generate dataset
            if use_web_search:
                # Generate directly using web search
                df, dataset_info = generate_dataset(
                    prompt=prompt,
                    num_rows=num_rows,
                    columns=columns,
                    model_name=model_name,
                    schema=schema,
                    use_web_search=True
                )
                
                # Generate visualizations
                charts = generate_visualizations(df, dataset_info['id'])
                
                # Store charts information
                dataset_info['charts'] = charts
                
                # Store dataset ID in session
                session['current_dataset_id'] = dataset_info['id']
                
                # Preview data
                preview = df.head(10).to_html(classes='table table-striped', index=False)
                
                # Add JavaScript for saving to localStorage
                client_side_data = {
                    'id': dataset_info['id'],
                    'name': dataset_info['name'],
                    'description': dataset_info['description'],
                    'columns': dataset_info['columns'],
                    'row_count': dataset_info['row_count'],
                    'created_at': dataset_info['created_at'],
                    'model_used': dataset_info['model_used'],
                    'web_search_used': dataset_info['web_search_used'],
                    'format': dataset_info['format']
                }
                
                return render_template('result.html',
                                     preview=preview,
                                     total_rows=dataset_info['row_count'],
                                     filename=dataset_info['file_name'],
                                     dataset=dataset_info,
                                     dataset_json=json.dumps(client_side_data),
                                     theme=theme)
            else:
                # Generate initial dataset with AI only
                csv_data, _ = generate_dataset(prompt, num_rows, columns, model_name, schema)
                
                # Process the data
                df = process_csv_data(csv_data, columns)
                
                # Check if we need to continue generating (if not enough rows)
                attempts = 1
                max_attempts = 5  # Prevent infinite loops
                
                while len(df) < num_rows and attempts < max_attempts:
                    # Summarize what we have so far
                    summary = df.head(5).to_csv(index=False)
                    if len(df) > 5:
                        summary += f"\n... and {len(df) - 5} more rows"
                    
                    # Generate more data
                    more_csv_data, _ = generate_dataset(prompt, num_rows - len(df), columns, model_name, schema, summary)
                    
                    # Process additional data
                    more_df = process_csv_data(more_csv_data, columns)
                    
                    # Append new data
                    df = pd.concat([df, more_df], ignore_index=True)
                    
                    # Limit to requested number of rows
                    if len(df) > num_rows:
                        df = df.iloc[:num_rows]
                        
                    attempts += 1
                
                # Apply missing values if requested
                if missing_values_pct > 0:
                    # Calculate how many cells should be missing
                    total_cells = df.shape[0] * df.shape[1]
                    cells_to_nullify = int(total_cells * missing_values_pct / 100)
                    
                    # Randomly set values to NaN
                    import numpy as np
                    rows = np.random.randint(0, df.shape[0], size=cells_to_nullify)
                    cols = np.random.randint(0, df.shape[1], size=cells_to_nullify)
                    for r, c in zip(rows, cols):
                        df.iloc[r, c] = np.nan
                
                # Generate unique dataset ID
                global last_dataset_id
                last_dataset_id += 1
                dataset_id = last_dataset_id
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Save to file based on format
                if output_format == 'csv':
                    filename = f"dataset_{dataset_id}_{timestamp}.csv"
                    file_path = os.path.join(UPLOAD_FOLDER, filename)
                    df.to_csv(file_path, index=False)
                elif output_format == 'json':
                    filename = f"dataset_{dataset_id}_{timestamp}.json"
                    file_path = os.path.join(UPLOAD_FOLDER, filename)
                    df.to_json(file_path, orient='records')
                elif output_format == 'excel':
                    filename = f"dataset_{dataset_id}_{timestamp}.xlsx"
                    file_path = os.path.join(UPLOAD_FOLDER, filename)
                    df.to_excel(file_path, index=False)
                else:  # Default to CSV
                    filename = f"dataset_{dataset_id}_{timestamp}.csv"
                    file_path = os.path.join(UPLOAD_FOLDER, filename)
                    df.to_csv(file_path, index=False)
                
                # Generate visualizations
                charts = generate_visualizations(df, dataset_id)
                
                # Store dataset in our database
                dataset_info = {
                    'id': dataset_id,
                    'name': f"Dataset {dataset_id}",
                    'description': prompt[:100],
                    'columns': columns,
                    'schema': schema,
                    'row_count': len(df),
                    'file_path': file_path,
                    'file_name': filename,
                    'format': output_format,
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_used': model_name,
                    'charts': charts,
                    'web_search_used': False
                }
                datasets_db.append(dataset_info)
                
                # Store dataset info in session
                session['current_dataset_id'] = dataset_id
                
                # Preview data
                preview = df.head(10).to_html(classes='table table-striped', index=False)
                
                return render_template('result.html',
                                     preview=preview,
                                     total_rows=len(df),
                                     filename=filename,
                                     dataset=dataset_info,
                                     theme=theme)
                
        except Exception as e:
            flash(f"Error: {str(e)}")
            return redirect(url_for('index'))
    
    return render_template('index.html',
                           models=AVAILABLE_MODELS,
                           theme=theme)

@app.route('/theme/<theme_name>')
def set_theme(theme_name):
    # Validate theme
    if theme_name not in ['light', 'dark']:
        theme_name = 'light'
    
    # Set theme in session
    session['theme'] = theme_name
    
    # Redirect back to previous page or home
    return redirect(request.referrer or url_for('index'))

@app.route('/download/<int:dataset_id>')
def download(dataset_id):
    # Find dataset in our database
    dataset = next((d for d in datasets_db if d['id'] == dataset_id), None)
    
    if not dataset or not os.path.exists(dataset.get('file_path', '')):
        flash("Dataset not found. Please generate a new dataset.")
        return redirect(url_for('index'))
    
    file_path = dataset['file_path']
    filename = dataset['file_name']
    
    # Set mimetype based on format
    if dataset['format'] == 'csv':
        mimetype = 'text/csv'
    elif dataset['format'] == 'json':
        mimetype = 'application/json'
    elif dataset['format'] == 'excel':
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    else:
        mimetype = 'application/octet-stream'
    
    return send_file(file_path,
                   mimetype=mimetype,
                   download_name=filename,
                   as_attachment=True)

@app.route('/visualize/<int:dataset_id>')
def visualize(dataset_id):
    # Get theme preference
    theme = session.get('theme', 'light')
    
    # Find dataset in our database
    dataset = next((d for d in datasets_db if d['id'] == dataset_id), None)
    
    if not dataset or not os.path.exists(dataset.get('file_path', '')):
        flash("Dataset not found. Please generate a new dataset.")
        return redirect(url_for('index'))
    
    # Load the dataset
    if dataset['format'] == 'csv':
        df = pd.read_csv(dataset['file_path'])
    elif dataset['format'] == 'json':
        df = pd.read_json(dataset['file_path'])
    elif dataset['format'] == 'excel':
        df = pd.read_excel(dataset['file_path'])
    else:
        flash(f"Unsupported format for visualization: {dataset['format']}")
        return redirect(url_for('index'))
    
    # If we don't have charts, generate them now
    if 'charts' not in dataset or not dataset['charts']:
        dataset['charts'] = generate_visualizations(df, dataset_id)
    
    # Generate data quality report
    quality_stats = {}
    for col in df.columns:
        quality_stats[col] = {
            'missing': int(df[col].isna().sum()),
            'missing_percent': f"{(df[col].isna().sum() / len(df) * 100):.2f}%",
            'unique_values': int(df[col].nunique()),
            'top_value': str(df[col].mode().iloc[0]) if not df[col].mode().empty else 'N/A'
        }
    
    # Generate basic data info
    data_info = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isna().sum().sum(),
        'missing_percent': f"{(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%"
    }
    
    return render_template('visualize.html',
                          dataset=dataset,
                          data_info=data_info,
                          charts=dataset['charts'],
                          quality_stats=quality_stats,
                          theme=theme)

@app.route('/dashboard')
def dashboard():
    # Get theme preference
    theme = session.get('theme', 'light')
    
    # Get all datasets
    datasets = datasets_db
    
    # Calculate summary stats
    stats = {
        'total_datasets': len(datasets),
        'total_rows': sum(d.get('row_count', 0) for d in datasets),
        'models_used': {}
    }
    
    # Count usage by model
    for dataset in datasets:
        model = dataset.get('model_used', 'unknown')
        if model in stats['models_used']:
            stats['models_used'][model] += 1
        else:
            stats['models_used'][model] = 1
    
    return render_template('dashboard.html',
                          datasets=datasets,
                          stats=stats,
                          theme=theme)

@app.route('/export/<int:dataset_id>/<format>')
def export(dataset_id, format):
    # Find dataset in our database
    dataset = next((d for d in datasets_db if d['id'] == dataset_id), None)
    
    if not dataset or not os.path.exists(dataset.get('file_path', '')):
        flash("Dataset not found. Please generate a new dataset.")
        return redirect(url_for('index'))
    
    # Load the original data
    original_format = dataset['format']
    if original_format == 'csv':
        df = pd.read_csv(dataset['file_path'])
    elif original_format == 'json':
        df = pd.read_json(dataset['file_path'])
    elif original_format == 'excel':
        df = pd.read_excel(dataset['file_path'])
    else:
        flash(f"Unsupported format for conversion: {original_format}")
        return redirect(url_for('index'))
    
    # Create a new file in the requested format
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if format == 'csv':
        filename = f"dataset_{dataset_id}_{timestamp}.csv"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        df.to_csv(file_path, index=False)
        mimetype = 'text/csv'
    elif format == 'json':
        filename = f"dataset_{dataset_id}_{timestamp}.json"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        df.to_json(file_path, orient='records')
        mimetype = 'application/json'
    elif format == 'excel':
        filename = f"dataset_{dataset_id}_{timestamp}.xlsx"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        df.to_excel(file_path, index=False)
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    elif format == 'sqlite':
        filename = f"dataset_{dataset_id}_{timestamp}.db"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        from sqlalchemy import create_engine
        conn = create_engine(f'sqlite:///{file_path}')
        df.to_sql('dataset', conn, if_exists='replace', index=False)
        mimetype = 'application/x-sqlite3'
    else:
        flash(f"Unsupported export format: {format}")
        return redirect(url_for('index'))
    
    return send_file(file_path,
                   mimetype=mimetype,
                   download_name=filename,
                   as_attachment=True)



@app.route('/api/dataset/<int:dataset_id>')
def api_get_dataset(dataset_id):
    # Find dataset in our database
    dataset = next((d for d in datasets_db if d['id'] == dataset_id), None)
    
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    
    # Load the dataset
    try:
        if dataset['format'] == 'csv':
            df = pd.read_csv(dataset['file_path'])
        elif dataset['format'] == 'json':
            df = pd.read_json(dataset['file_path'])
        elif dataset['format'] == 'excel':
            df = pd.read_excel(dataset['file_path'])
        else:
            return jsonify({'error': f"Unsupported format: {dataset['format']}"}), 400
        
        # Convert to JSON
        data = df.to_dict(orient='records')
        
        # Include validation if available
        validation = dataset.get('validation', None)
        
        return jsonify({
            'dataset': {
                'id': dataset['id'],
                'name': dataset['name'],
                'description': dataset['description'],
                'row_count': dataset['row_count'],
                'created_at': dataset['created_at'],
                'format': dataset['format'],
                'web_search_used': dataset.get('web_search_used', False),
                'validation': validation
            },
            'data': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Update the index.html template to include web search option
@app.route('/update_index_template')
def update_index_template():
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'index.html')
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Check if web search option already exists
    if 'use_web_search' in content:
        return jsonify({'message': 'Template already updated'})
    
    # Add web search option to the advanced options section
    web_search_html = """
    <div class="form-check mb-3">
        <input type="checkbox" class="form-check-input" id="use-web-search" name="use_web_search">
        <label class="form-check-label" for="use-web-search">Use web search to enhance dataset (DuckDuckGo)</label>
        <div class="form-text">Search the web to find relevant information for your dataset.</div>
    </div>
    """
    
    # Insert before the closing div of advanced options
    updated_content = content.replace('</div>\n            \n            <div class="d-grid gap-2">', web_search_html + '</div>\n            \n            <div class="d-grid gap-2">')
    
    with open(template_path, 'w') as f:
        f.write(updated_content)
    
    return jsonify({'message': 'Template updated successfully'})

if __name__ == '__main__':
    try:
        # First update the template to include web search option
        update_url = 'http://localhost:5000/update_index_template'
        print(f"You may need to visit {update_url} once to update the template.")
        
        # Turn on debug mode for Flask
        app.debug = True
        
        # Print confirmation to verify code execution
        print("Starting Gemini Dataset Generator with Web Search capabilities...")
        print(f"Available models: {list(AVAILABLE_MODELS.keys())}")
        print(f"Templates directory: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')}")
        
        # Run the app
        app.run(debug=True)
    except Exception as e:
        # Print any exception that occurs
        import traceback
        print(f"ERROR STARTING APPLICATION: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
