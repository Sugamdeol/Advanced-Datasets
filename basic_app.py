import os
import csv
import tempfile
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash, send_file, session
import google.generativeai as genai
from io import StringIO

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure Gemini API with the provided key
GEMINI_API_KEY = "AIzaSyB_dNcoAstUzFWW3c_N5vHtl08YfSYSxOA"
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

def generate_dataset(prompt, num_rows, columns, previous_data=None):
    """Generate dataset using Gemini API"""
    try:
        # Prepare the prompt for Gemini
        full_prompt = f"""
        Generate a high-quality dataset with {num_rows} rows and the following columns: {', '.join(columns)}.
        
        Format the response as CSV data without markdown formatting.
        Do not include any explanations, just return raw CSV data.
        
        Dataset description: {prompt}
        """
        
        # If continuing from previous data, include that context
        if previous_data:
            full_prompt += f"\n\nThis is a continuation. Here's what has been generated so far (summary):\n{previous_data}\nPlease continue from where this left off, maintaining consistency with previous data."
        
        # Generate content from Gemini
        response = model.generate_content(full_prompt)
        
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

# Available models that can be used
AVAILABLE_MODELS = {
    'gemini-2.5-flash-preview-04-17': 'Gemini 2.5 Flash (Default)',
    'gemini-1.5-flash': 'Gemini 1.5 Flash',
    'gemini-1.5-pro': 'Gemini 1.5 Pro'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    # Set default theme to light
    theme = session.get('theme', 'light')
    
    if request.method == 'POST':
        prompt = request.form.get('prompt', '')
        columns_input = request.form.get('columns', '')
        num_rows = int(request.form.get('num_rows', 10))
        
        if not prompt or not columns_input:
            flash("Please provide both a prompt and columns")
            return redirect(url_for('index'))
            
        # Parse columns
        columns = [col.strip() for col in columns_input.split(',')]
        
        try:
            # Generate initial dataset
            csv_data = generate_dataset(prompt, num_rows, columns)
            
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
                more_csv_data = generate_dataset(prompt, num_rows - len(df), columns, summary)
                
                # Process additional data
                more_df = process_csv_data(more_csv_data, columns)
                
                # Append new data
                df = pd.concat([df, more_df], ignore_index=True)
                
                # Limit to requested number of rows
                if len(df) > num_rows:
                    df = df.iloc[:num_rows]
                    
                attempts += 1
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            df.to_csv(temp_file.name, index=False)
            temp_file.close()
            
            # Store file path in session
            session['csv_file_path'] = temp_file.name
            session['csv_filename'] = f"dataset_{prompt[:20].replace(' ', '_')}.csv"
            
            # Preview data
            preview = df.head(10).to_html(classes='table table-striped', index=False)
            total_rows = len(df)
            
            # Create dataset info dictionary for the template
            dataset_info = {
                'id': 1,
                'name': f"Generated Dataset",
                'description': prompt[:100],
                'columns': columns,
                'row_count': len(df),
                'file_name': session['csv_filename'],
                'format': 'csv',
                'created_at': 'Just now',
                'model_used': 'gemini-2.5-flash-preview-04-17'
            }
            
            return render_template('result.html', 
                                 preview=preview, 
                                 total_rows=total_rows,
                                 filename=session['csv_filename'],
                                 dataset=dataset_info,
                                 theme=theme)
            
        except Exception as e:
            flash(f"Error: {str(e)}")
            return redirect(url_for('index'))
    
    return render_template('index.html', models=AVAILABLE_MODELS, theme=theme)

@app.route('/theme/<theme_name>')
def set_theme(theme_name):
    # Validate theme
    if theme_name not in ['light', 'dark']:
        theme_name = 'light'
    
    # Set theme in session
    session['theme'] = theme_name
    
    # Redirect back to previous page or home
    return redirect(request.referrer or url_for('index'))

@app.route('/download')
def download():
    if 'csv_file_path' in session and 'csv_filename' in session:
        file_path = session['csv_file_path']
        filename = session['csv_filename']
        
        if os.path.exists(file_path):
            return send_file(file_path, 
                           mimetype='text/csv',
                           download_name=filename,
                           as_attachment=True)
        else:
            flash("File not found. Please generate a new dataset.")
    else:
        flash("No dataset found. Please generate a dataset first.")
        
    return redirect(url_for('index'))

@app.route('/download/<int:dataset_id>')
def download_by_id(dataset_id):
    # In the basic app, we only have one dataset, so we ignore the ID
    # and just serve the file in the session
    return download()

@app.route('/visualize/<int:dataset_id>')
def visualize(dataset_id):
    # In the basic app, we don't have full visualization features
    # Just redirect to home and show a message
    flash("Visualization features are only available in the enhanced version.")
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    # Simple dashboard for the basic app
    flash("Dashboard features are only available in the enhanced version.")
    return redirect(url_for('index'))

@app.route('/templates')
def templates_list():
    # Templates page for the basic app
    flash("Templates system is only available in the enhanced version.")
    return redirect(url_for('index'))

@app.route('/api/dataset/<int:dataset_id>')
def api_get_dataset(dataset_id):
    # Simple API endpoint for the basic app
    from flask import jsonify
    if 'csv_file_path' in session:
        try:
            df = pd.read_csv(session['csv_file_path'])
            return jsonify({
                'data': df.to_dict(orient='records')
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'No dataset found'}), 404

@app.route('/export/<int:dataset_id>/<format>')
def export(dataset_id, format):
    # In the basic app, we only support CSV export
    flash("Multiple export formats are only available in the enhanced version.")
    return download()

if __name__ == '__main__':
    app.run(debug=True)
