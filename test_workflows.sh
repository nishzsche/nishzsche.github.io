# Define the final output folder
final_folder="_posts"

# Ensure the output folder exists
mkdir -p "$final_folder"
for notebook in $(find notebooks -name '*.ipynb'); do
    echo "Processing $notebook"

    # Convert the Jupyter notebook to Markdown
    jupyter nbconvert --to markdown "$notebook"

    # Determine the generated Markdown filename
    markdown_file="${notebook%.ipynb}.md"

    # Normalize the path: Replace backslashes (\) with forward slashes (/)
    markdown_file=$(echo "$markdown_file" | sed 's/\\/\//g')

    # Ensure the markdown file exists
    if [ ! -f "$markdown_file" ]; then
        echo "Error: Markdown conversion failed for $notebook"
        exit 1
    fi

    # Extract the first heading from the converted Markdown file
    title=$(grep -m 1 '^# ' "$markdown_file" | sed 's/^# //')

    # If no title was found, set a default title based on the filename
    if [ -z "$title" ]; then
        title=$(basename "$notebook" .ipynb)
    fi

    # Sanitize the notebook filename by replacing special characters
    sanitized_filename=$(basename "$notebook" .ipynb | sed 's/[^a-zA-Z0-9]/_/g')

    # Define the output folder (replace backslashes with forward slashes if needed)
    output_file="$final_folder/${sanitized_filename}.md"

    # Print the sanitized filename and normalized paths for debugging
    echo "Sanitized filename: $sanitized_filename"
    echo "Markdown file path (normalized): $markdown_file"

    # Add the full markdown heading to the file with the extracted title
    echo -e "---\nlayout: post\ntitle: \"$title\"\n---\n\n$(cat "$markdown_file")" > "$output_file"

    # Print the content of the final Markdown file for debugging
    echo "Final Markdown file content:"
    cat "$output_file"
done
