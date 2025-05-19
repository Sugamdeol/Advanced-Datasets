import os
import csv
import json
import tempfile
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, flash, send_file, session, jsonify
import google.generativeai as genai
from io import StringIO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure Gemini API with the provided key
GEMINI_API_KEY = "AIzaSyB_dNcoAstUzFWW3c_N5vHtl08YfSYSxOA"
genai.configure(api_key=GEMINI_API_KEY)

# Available models
AVAILABLE_MODELS = {
    'gemini-2.5-flash-preview-04-17': 'Gemini 2.5 Flash (Default)',
    'gemini-1.5-flash': 'Gemini 1.5 Flash',
    'gemini-1.5-pro': 'Gemini 1.5 Pro',
    'gemini-1.0-pro': 'Gemini 1.0 Pro'
}

# Initialize default Gemini model
DEFAULT_MODEL = 'gemini-2.5-flash-preview-04-17'
model = genai.GenerativeModel(DEFAULT_MODEL)

# Create upload directory if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database of generated datasets (in-memory for simplicity)
datasets_db = []

def generate_dataset(prompt, num_rows, columns, model_name=DEFAULT_MODEL, schema=None, previous_data=None):
    """Generate dataset using Gemini API"""
    try:
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
            
        return csv_data
    except Exception as e:
        return f"Error generating dataset: {str(e)}"

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

@app.route('/', methods=['GET', 'POST'])
def index():
    # Get theme preference from session or default to light
    theme = session.get('theme', 'light')
    
    if request.method == 'POST':
        prompt = request.form.get('prompt', '')
        columns_input = request.form.get('columns', '')
        num_rows = int(request.form.get('num_rows', 50))
        model_name = request.form.get('model', DEFAULT_MODEL)
        output_format = request.form.get('output_format', 'csv')
        
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
            # Generate initial dataset
            csv_data = generate_dataset(prompt, num_rows, columns, model_name, schema)
            
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
                more_csv_data = generate_dataset(prompt, num_rows - len(df), columns, model_name, schema, summary)
                
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
            dataset_id = len(datasets_db) + 1
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
                'model_used': model_name
            }
            datasets_db.append(dataset_info)
            
            # Store dataset info in session
            session['current_dataset_id'] = dataset_id
            
            # Preview data
            preview = df.head(10).to_html(classes='table table-striped', index=False)
            total_rows = len(df)
            
            return render_template('result.html', 
                                  preview=preview, 
                                  total_rows=total_rows,
                                  filename=filename,
                                  dataset=dataset_info,
                                  theme=theme)
            
        except Exception as e:
            flash(f"Error: {str(e)}")
            return redirect(url_for('index'))
    
    return render_template('index.html', 
                           models=AVAILABLE_MODELS,
                           theme=theme)

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
    
    # Create visualization directory if it doesn't exist
    vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Generate visualizations
    visualizations = []
    
    # 1. Data Overview
    data_info = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isna().sum().sum(),
        'missing_percent': f"{(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%"
    }
    
    # 2. Get column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    # 3. Generate different chart types based on column types
    charts = []
    
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
    
    # Generate data quality report
    quality_stats = {}
    for col in df.columns:
        quality_stats[col] = {
            'missing': int(df[col].isna().sum()),
            'missing_percent': f"{(df[col].isna().sum() / len(df) * 100):.2f}%",
            'unique_values': int(df[col].nunique()),
            'top_value': str(df[col].mode().iloc[0]) if not df[col].mode().empty else 'N/A'
        }
    
    return render_template('visualize.html',
                          dataset=dataset,
                          data_info=data_info,
                          charts=charts,
                          quality_stats=quality_stats,
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

@app.route('/templates')
def templates_list():
    # Sample templates - in a real app, these would come from a database
    templates = [
        {
            'id': 1,
            'name': 'Customer Database',
            'description': 'Customer data with demographics and purchase history',
            'columns': ['id', 'name', 'email', 'age', 'location', 'signup_date', 'total_purchases', 'last_purchase_date']
        },
        {
            'id': 2,
            'name': 'Product Catalog',
            'description': 'Product listing with details and inventory',
            'columns': ['product_id', 'name', 'description', 'category', 'price', 'inventory_count', 'supplier', 'last_restocked']
        },
        {
            'id': 3,
            'name': 'Financial Transactions',
            'description': 'Financial transaction records with categories and amounts',
            'columns': ['transaction_id', 'date', 'amount', 'category', 'description', 'account', 'balance_after', 'is_credit']
        }
    ]
    
    # Get theme preference
    theme = session.get('theme', 'light')
    
    return render_template('templates.html',
                          templates=templates,
                          theme=theme)

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
        
        return jsonify({
            'dataset': {
                'id': dataset['id'],
                'name': dataset['name'],
                'description': dataset['description'],
                'row_count': dataset['row_count'],
                'created_at': dataset['created_at'],
                'format': dataset['format']
            },
            'data': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Create a directory for static files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(static_dir, exist_ok=True)

# Create a directory for visualizations
vis_dir = os.path.join(static_dir, 'visualizations')
os.makedirs(vis_dir, exist_ok=True)

if __name__ == '__main__':
    app.run(debug=True)
