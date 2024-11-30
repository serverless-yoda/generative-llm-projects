import os
from typing import List, Dict, Any
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from dotenv import load_dotenv
from functools import partial

# Load environment variables
load_dotenv()

class DatabaseComparison:
    def __init__(self):
        """
        Initialize database connections for Databricks and SQL Server
        """
        
        databricks_connection_string = os.environ['DATABRICKS']
        self.databricks_engine = create_engine(databricks_connection_string)

        # SQL Server connection
        sqlserver_server = os.getenv('SQLSERVER_SERVER')
        sqlserver_database = os.getenv('SQLSERVER_DATABASE')
        sqlserver_username = os.getenv('SQLSERVER_USERNAME')
        sqlserver_password = os.getenv('SQLSERVER_PASSWORD')

        # SQLAlchemy connection string for SQL Server
        sqlserver_connection_string = (
            f"mssql+pyodbc://{sqlserver_username}:{sqlserver_password}@"
            f"{sqlserver_server}/{sqlserver_database}?driver=ODBC+Driver+17+for+SQL+Server"
        )
        self.sqlserver_engine = create_engine(sqlserver_connection_string)

    def fetch_databricks_data(self, query: str) -> pd.DataFrame:
        """
        Fetch data from Databricks using a SQL query
        
        :param query: SQL query to execute
        :return: Pandas DataFrame with query results
        """
        return pd.read_sql(query, self.databricks_engine)

    def fetch_sqlserver_data(self, query: str) -> pd.DataFrame:
        """
        Fetch data from SQL Server using a SQL query
        
        :param query: SQL query to execute
        :return: Pandas DataFrame with query results
        """
        return pd.read_sql(query, self.sqlserver_engine)

    
    def compare_dataframes(
        self, 
        databricks_df: pd.DataFrame, 
        sqlserver_df: pd.DataFrame, 
        key_columns: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Compare two dataframes and identify missing records
        
        :param databricks_df: DataFrame from Databricks
        :param sqlserver_df: DataFrame from SQL Server
        :param key_columns: List of column names to use as primary key
        :return: Dictionary with missing records
        """
        # Create composite key for both dataframes
        databricks_df['composite_key'] = databricks_df[key_columns].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        print('databricks done')
        sqlserver_df['composite_key'] = sqlserver_df[key_columns].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        print('sqlserver done')

        # Find records in Databricks missing from SQL Server
        missing_from_sqlserver = databricks_df[~databricks_df['composite_key'].isin(sqlserver_df['composite_key'])]
        
        # Find records in SQL Server missing from Databricks
        missing_from_databricks = sqlserver_df[~sqlserver_df['composite_key'].isin(databricks_df['composite_key'])]

        # Drop the composite key columns
        missing_from_sqlserver = missing_from_sqlserver.drop(columns=['composite_key'])
        missing_from_databricks = missing_from_databricks.drop(columns=['composite_key'])

        return {
            'missing_from_sqlserver': missing_from_sqlserver,
            'missing_from_databricks': missing_from_databricks
        }

    def main(self,query):
        """
        Main method to demonstrate the comparison process
        """
        try:
            # Example queries - replace with your actual table/query
            databricks_query = query #f"select reference,count(reference) as count_reference from stage_BB_order_1_travel group by reference order by count_reference desc"
            sqlserver_query = query  #f"select reference,count(reference) as count_reference from stage_BB_order_1_travel group by reference order by count_reference desc"

            # Fetch data
            databricks_data = self.fetch_databricks_data(databricks_query)
            print('get databricks test data')
            print(databricks_data.head())
            sqlserver_data = self.fetch_sqlserver_data(sqlserver_query)
            print('get sqlserver test data')
            print(sqlserver_data.head())

            # Define key columns for comparison
            key_columns = ['reference','count_reference']  # Replace with your actual key columns

            # Compare dataframes
            comparison_result = self.compare_dataframes(
                databricks_data, 
                sqlserver_data, 
                key_columns
            )

            # Print missing records
            print("Records missing from SQL Server:")
            print(comparison_result['missing_from_sqlserver'])
            
            print("\nRecords missing from Databricks:")
            print(comparison_result['missing_from_databricks'])

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    comparison = DatabaseComparison()
    query = f"select reference,count(reference) as count_reference from stage_test_table group by reference order by count_reference desc"
   
 
    comparison.main(query)
