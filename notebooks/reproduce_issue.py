
import pandas as pd
import os

# Set working directory to notebooks loc
os.chdir(r"c:\Users\Iman\OneDrive\Desktop\knowledge_graph_project\notebooks")

try:
    print("Loading Edges...")
    df_edges_plant = pd.read_csv("../data/SupplyGraph/Edges/Edges (Plant).csv")
    print("Edges (Plant) loaded.")
    df_edges_product_group = pd.read_csv("../data/SupplyGraph/Edges/Edges (Product Group).csv")
    print("Edges (Product Group) loaded.")
    df_edges_product_subgroup = pd.read_csv("../data/SupplyGraph/Edges/Edges (Product Sub-Group).csv")
    print("Edges (Product Sub-Group) loaded.")
    df_edges_storage_location = pd.read_csv("../data/SupplyGraph/Edges/Edges (Storage Location).csv")
    print("Edges (Storage Location) loaded.")

    print("Loading Nodes...")
    df_nodes_productgroup_and_subgroup = pd.read_csv("../data/SupplyGraph/Nodes/Node Types (Product Group and Subgroup).csv")
    print("Nodes (Product Group and Subgroup) loaded.")
    df_nodes_plant_and_storage = pd.read_csv("../data/SupplyGraph/Nodes/Nodes Type (Plant & Storage).csv")
    print("Nodes (Plant & Storage) loaded.")
    df_nodes = pd.read_csv("../data/SupplyGraph/Nodes/Nodes.csv")
    print("Nodes loaded.")

    print("Loading Temporal...")
    df_temporal_delivery_to_distributor = pd.read_csv("../data/SupplyGraph/Temporal Data/Unit/Delivery To distributor.csv")
    print("Delivery loaded.")
    df_temporal_factory_issue = pd.read_csv("../data/SupplyGraph/Temporal Data/Unit/Factory Issue.csv")
    print("Factory Issue loaded.")
    df_temporal_production = pd.read_csv("../data/SupplyGraph/Temporal Data/Unit/Production.csv")
    print("Production loaded.")
    df_temporal_sales_order = pd.read_csv("../data/SupplyGraph/Temporal Data/Unit/Sales Order.csv")
    print("Sales Order loaded.")

    print("Pivoting datasets...")
    df_temporal_sales_order['Date'] = pd.to_datetime(df_temporal_sales_order['Date'])
    df_temporal_sales_order_pivot = df_temporal_sales_order.melt(id_vars='Date', var_name='Product', value_name='Sales').dropna()
    print("Sales Order pivoted.")

    df_temporal_production['Date'] = pd.to_datetime(df_temporal_production['Date'])
    df_temporal_production_pivot = df_temporal_production.melt(id_vars='Date', var_name='Product', value_name='Production Quantity').dropna()
    print("Production pivoted.")

    df_temporal_factory_issue['Date'] = pd.to_datetime(df_temporal_factory_issue['Date'])
    df_temporal_factory_issue_pivot = df_temporal_factory_issue.melt(id_vars='Date', var_name='Product', value_name='Factory Issue').dropna()
    print("Factory Issue pivoted.")

    df_temporal_delivery_to_distributor['Date'] = pd.to_datetime(df_temporal_delivery_to_distributor['Date'])
    df_temporal_delivery_to_distributor_pivot = df_temporal_delivery_to_distributor.melt(id_vars='Date', var_name='Product', value_name='Distributor').dropna()
    print("Delivery pivoted.")

    print("All success.")

except Exception as e:
    print(f"FAILED with error: {e}")
