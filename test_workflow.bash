# Define the final output folder
final_folder="_posts"
mkdir -p "$final_folder"
for notebook in $(find notebooks -name '*.ipynb'); do
    # Sanitize the notebook filename by replacing special characters
    sanitized_filename=$(basename "$notebook" .ipynb | sed 's/[^a-zA-Z0-9]/_/g')

    # Define the output markdown file path
    output_file="$final_folder/${sanitized_filename}.md"

    # Check if the Markdown file already exists, skip conversion if it does
    if [ -f "$output_file" ]; then
        echo "Skipping $notebook, corresponding Markdown file already exists."
        continue
    fi

    echo "Processing $notebook"
    jq -M 'del(.metadata.widgets)' $notebook > $sanitized_filename.ipynb
    # Convert the Jupyter notebook to Markdown
    jupyter nbconvert --to markdown "$sanitized_filename.ipynb"
    
    # Determine the generated Markdown filename
    markdown_file="${sanitized_filename%.ipynb}.md"

    echo markdown_file
done