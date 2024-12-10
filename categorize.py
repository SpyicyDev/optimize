import pandas as pd

def categorize_stocks(master_list, file_path, sheet_name='Sheet1', column_name='Ticker'):
    """
    Categorize stocks into on_list and off_list based on their presence in an Excel file.

    Args:
        master_list (list): List of master stock tickers to categorize.
        file_path (str): Path to the Excel file.
        sheet_name (str): Name of the Excel sheet containing the stock list.
        column_name (str): Name of the column containing stock tickers in the Excel sheet.

    Returns:
        tuple: Two lists - on_list and off_list.
    """
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Ensure the column name matches
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the Excel sheet.")

    # Extract the stock tickers from the Excel file
    excel_stocks = df[column_name].dropna().astype(str).tolist()

    # Categorize stocks
    on_list = [stock for stock in master_list if stock in excel_stocks]
    off_list = [stock for stock in master_list if stock not in excel_stocks]

    return on_list, off_list

if __name__ == "__main__":
    # Updated master list of stocks
    master_list = [
        'ASC', 'COST', 'FDX', 'GMAB', 'GSK', 'LZRFY', 'MA', 'MTNOY',
        'RIVN', 'SIEGY', 'SPY', 'TGT', 'TSLA', 'TSM', 'TTE', '500440',
        'CRPG5', 'SEPL'
    ]

    # Path to the Excel file
    file_path = 'stock_list.xlsx'

    # Call the function and categorize stocks
    try:
        on_list, off_list = categorize_stocks(master_list, file_path, sheet_name='Sheet1', column_name='Ticker')
        print("on_list:", on_list)
        print("off_list:", off_list)
    except Exception as e:
        print(f"An error occurred: {e}")