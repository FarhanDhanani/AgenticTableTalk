import json

def copy_tables(reference_file_path, target_file_path, output_file_path):
    # Load reference tables
    with open(reference_file_path, 'r') as ref_file:
        reference_data = json.load(ref_file)

    # Flatten reference_data into a dictionary: { "table0": [...], "table1": [...] }
    reference_tables = {}
    for item in reference_data:
        reference_tables.update(item)

    # Load target file
    with open(target_file_path, 'r') as target_file:
        target_data = json.load(target_file)

    # Update target_data with matching tables from reference
    for entry in target_data:
        table_id = entry.get('table_id')
        if table_id in reference_tables:
            entry['table'] = reference_tables[table_id]

    # Save the updated target data to a new file
    with open(output_file_path, 'w') as output_file:
        json.dump(target_data, output_file, indent=2)

    print(f"Updated tables written to {output_file_path}")

def reverse_copy_tables(reference_file_path, target_file_path, output_file_path):
    """
    Copies updated tables from `target_file_path` into `reference_file_path`
    by matching on table_id, and writes the merged result to `output_file_path`.

    This is the reverse of `copy_tables()` — it replaces the reference's tables
    with the updated ones found in the target.
    """
    # Load target file
    with open(target_file_path, 'r') as target_file:
        target_data = json.load(target_file)

    # Build a lookup: { "table0": [table_data], ... }
    target_tables = {
        entry["table_id"]: entry["table"]
        for entry in target_data
        if "table_id" in entry and "table" in entry
    }

    # Load reference file
    with open(reference_file_path, 'r') as ref_file:
        reference_data = json.load(ref_file)

    # Update reference tables using the lookup from target
    for table_obj in reference_data:
        for table_id in table_obj:
            if table_id in target_tables:
                table_obj[table_id] = target_tables[table_id]

    # Write the updated reference file
    with open(output_file_path, 'w') as output_file:
        json.dump(reference_data, output_file, indent=2)

    print(f"Reversed table update written to {output_file_path}")

def filter_out_irrelevant_tables(input_file_path):
    with open(input_file_path, 'r') as target_file:
        target_data = json.load(target_file)
    
    top_10 = target_data[:10]

    if input_file_path:
        with open(input_file_path, 'w') as f:
            json.dump(top_10, f, indent=2)
    return 


# Example usage
#filter_out_irrelevant_tables('up_generated_large_table_questions.json')

reverse_copy_tables('generated_large_tables.json', 'up_generated_large_table_questions.json', 'up_large_table_questions.json')


