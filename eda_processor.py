import pandas as pd
import sqlite3
import os
import matplotlib.pyplot as plt
import seaborn as sns

class RPADataEDA:
    def __init__(self, db_path, output_db_name="eda_results.db"):
        # Ajustamos para que reconozca la estructura data/database/
        self.db_path = db_path
        self.base_dir = os.path.dirname(db_path)
        self.output_folder = os.path.join(self.base_dir, "Reports")
        self.output_db_path = os.path.join(self.base_dir, output_db_name)
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
    def _get_connection(self, path):
        return sqlite3.connect(path)

    def run_eda(self):
        with self._get_connection(self.db_path) as conn_in, \
             self._get_connection(self.output_db_path) as conn_out:
            
            tables = [row[0] for row in conn_in.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
            
            if not tables:
                print("⚠️ No se encontraron tablas en la base de datos.")
                return

            for table in tables:
                print(f"📊 Procesando tabla: {table}...")
                df = pd.read_sql(f"SELECT * FROM {table}", conn_in)
                
                # Guardar estadísticas en la nueva DB
                df.describe(include='all').transpose().to_sql(f"eda_{table}_stats", conn_out, if_exists='replace')
                
                # Generar visualizaciones
                numeric_cols = df.select_dtypes(include=['number']).columns
                if not numeric_cols.empty:
                    # Boxplot de Outliers
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=df[numeric_cols])
                    plt.title(f"Outliers - {table}")
                    plt.savefig(f"{self.output_folder}/{table}_outliers.png")
                    plt.close()
                
                print(f"✅ Análisis de {table} completado.")

if __name__ == "__main__":
    # RUTA ACTUALIZADA según tu imagen
    DB_INPUT = os.path.join("data", "database", "Procesos.db")
    
    if os.path.exists(DB_INPUT):
        print(f"📂 Archivo encontrado en: {DB_INPUT}")
        processor = RPADataEDA(DB_INPUT)
        processor.run_eda()
        print(f"\n🚀 Proceso finalizado. Resultados en: {os.path.join('data', 'database')}")
    else:
        print(f"❌ Error: No se encontró el archivo en {DB_INPUT}")
        print("Asegúrate de estar ejecutando el comando desde la raíz de tu proyecto.")