import numpy as np

def insert_dividers(matrix, context_size, row_divider_width, col_divider_width):
    total_rows, total_cols = matrix.shape

    # Insert row dividers
    insert_positions = np.arange(context_size, total_rows, context_size)
    insert_positions = np.repeat(insert_positions, row_divider_width)     # repeat the insert positions for each row divider
    matrix = np.insert(matrix, insert_positions, np.nan, axis=0)

    # Update total_rows and total_cols after row dividers insertion
    total_rows, total_cols = matrix.shape

    # Insert column dividers
    insert_positions = np.arange(context_size, total_cols, context_size)
    insert_positions = np.repeat(insert_positions, col_divider_width)     # repeat the insert positions for each column divider
    matrix = np.insert(matrix, insert_positions, np.nan, axis=1)

    return matrix

context_size = 2  # Example context size
dim_y = 8
row_divider_width = 5  # Example row divider width
col_divider_width = 2  # Example column divider width


att = np.arange(context_size * dim_y, dtype=np.float32)
att = np.tile(att, (context_size * dim_y, 1))

# Call the function
modified_matrix = insert_dividers(att, context_size, row_divider_width, col_divider_width)
modified_matrix.shape  # Check the new shape of the matrix
