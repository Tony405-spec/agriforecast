import sqlite3
import pandas as pd
import streamlit as st
import os
import time
import threading
import numpy as np
from datetime import datetime

class DatabaseManager:
    def __init__(self):
        self.db_path = self._get_database_path()
        self._write_lock = threading.Lock()
        self._initialize_tables()
    
    def _get_database_path(self):
        """Get database path that works on both local and Streamlit Cloud"""
        try:
            # For Streamlit Cloud, use /tmp directory which is writable
            if os.environ.get('STREAMLIT_SHARING') is not None or os.path.exists('/tmp'):
                return "/tmp/agripredict_ml.db"
            
            # Local development
            return "agripredict_ml.db"
            
        except Exception:
            return "agripredict_ml.db"
    
    def get_connection(self):
        """Get database connection with proper settings"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            return conn
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            return None
    
    def _initialize_tables(self):
        """Initialize tables with synchronized schema matching ML model features"""
        conn = self.get_connection()
        if conn:
            try:
                cursor = conn.cursor()
                
                # Users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        user_type TEXT CHECK(user_type IN ('farmer', 'extension_officer', 'researcher')),
                        county TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1
                    )
                """)
                
                # Predictions table - EXACT mapping to ML model features
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        username TEXT,
                        
                        -- ML Model Input Features (EXACT field names)
                        crop_type TEXT NOT NULL CHECK(crop_type IN ('Maize', 'Beans', 'Sorghum', 'Wheat', 'Millet')),
                        county TEXT NOT NULL,
                        soil_ph REAL NOT NULL CHECK(soil_ph >= 4.0 AND soil_ph <= 9.0),
                        soil_moisture REAL NOT NULL CHECK(soil_moisture >= 5.0 AND soil_moisture <= 50.0),
                        fertilizer_usage REAL NOT NULL CHECK(fertilizer_usage >= 0.0 AND fertilizer_usage <= 300.0),
                        temperature REAL NOT NULL CHECK(temperature >= 10.0 AND temperature <= 40.0),
                        rainfall REAL NOT NULL CHECK(rainfall >= 0.0 AND rainfall <= 300.0),
                        altitude REAL NOT NULL CHECK(altitude >= 0.0 AND altitude <= 2500.0),
                        
                        -- ML Model Output Results
                        predicted_yield REAL NOT NULL CHECK(predicted_yield >= 0),
                        confidence_lower REAL NOT NULL CHECK(confidence_lower >= 0),
                        confidence_upper REAL NOT NULL CHECK(confidence_upper >= 0),
                        std_dev REAL NOT NULL CHECK(std_dev >= 0),
                        
                        -- Additional metadata
                        soil_type TEXT CHECK(soil_type IN ('Clay', 'Loam', 'Sandy', 'Silt', 'Volcanic')),
                        humidity REAL,
                        model_version TEXT DEFAULT 'v1.0',
                        
                        -- Timestamps
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        FOREIGN KEY (user_id) REFERENCES users(id),
                        CHECK (confidence_lower <= predicted_yield AND predicted_yield <= confidence_upper)
                    )
                """)
                
                # Analytics table for aggregated insights
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        crop_type TEXT NOT NULL,
                        county TEXT NOT NULL,
                        
                        -- Performance metrics
                        average_yield REAL NOT NULL,
                        prediction_count INTEGER NOT NULL CHECK(prediction_count >= 0),
                        total_yield REAL NOT NULL,
                        min_yield REAL NOT NULL,
                        max_yield REAL NOT NULL,
                        yield_std_dev REAL NOT NULL,
                        
                        -- Feature averages for analysis
                        avg_soil_ph REAL,
                        avg_soil_moisture REAL,
                        avg_fertilizer_usage REAL,
                        avg_temperature REAL,
                        avg_rainfall REAL,
                        
                        -- Timestamps
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        UNIQUE(username, crop_type, county)
                    )
                """)
                
                # Model performance table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_version TEXT NOT NULL,
                        train_r2 REAL NOT NULL,
                        test_r2 REAL NOT NULL,
                        train_mae REAL NOT NULL,
                        test_mae REAL NOT NULL,
                        train_rmse REAL NOT NULL,
                        test_rmse REAL NOT NULL,
                        cv_scores_mean REAL NOT NULL,
                        cv_scores_std REAL NOT NULL,
                        feature_count INTEGER NOT NULL,
                        training_samples INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Feature importance table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS feature_importance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_version TEXT NOT NULL,
                        feature_name TEXT NOT NULL,
                        importance REAL NOT NULL,
                        rank INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(model_version, feature_name)
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_crop ON predictions(crop_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_county ON predictions(county)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(created_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_analytics_user_crop ON analytics(username, crop_type)")
                
                conn.commit()
                st.success("✅ ML Database tables initialized successfully!")
                
            except Exception as e:
                st.error(f"Error creating tables: {str(e)}")
            finally:
                conn.close()
    
    def store_user(self, user_data):
        """Store user data with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            conn = None
            try:
                with self._write_lock:
                    conn = self.get_connection()
                    if not conn:
                        return None
                    
                    cursor = conn.cursor()
                    
                    # Check if user exists
                    cursor.execute("SELECT id FROM users WHERE username = ?", (user_data['username'],))
                    existing_user = cursor.fetchone()
                    
                    if existing_user:
                        # Update existing user
                        cursor.execute("""
                            UPDATE users 
                            SET user_type = ?, county = ?, last_login = CURRENT_TIMESTAMP
                            WHERE username = ?
                        """, (user_data['user_type'], user_data['county'], user_data['username']))
                        user_id = existing_user[0]
                    else:
                        # Create new user
                        cursor.execute("""
                            INSERT INTO users (username, user_type, county, last_login)
                            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                        """, (
                            user_data['username'],
                            user_data['user_type'], 
                            user_data['county']
                        ))
                        user_id = cursor.lastrowid
                    
                    conn.commit()
                    return user_id
                    
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    st.error(f"Error storing user after {attempt + 1} attempts: {str(e)}")
                    return None
            except Exception as e:
                st.error(f"Error storing user: {str(e)}")
                return None
            finally:
                if conn:
                    conn.close()
        return None
    
    def store_prediction(self, user_data, input_data, prediction_result):
        """Store prediction with ML model features - EXACT field mapping"""
        max_retries = 3
        
        for attempt in range(max_retries):
            conn = None
            try:
                with self._write_lock:
                    conn = self.get_connection()
                    if not conn:
                        return False
                    
                    cursor = conn.cursor()
                    
                    # Insert prediction with EXACT ML model field names
                    cursor.execute("""
                        INSERT INTO predictions 
                        (user_id, username, crop_type, county, soil_ph, soil_moisture, 
                         fertilizer_usage, temperature, rainfall, altitude,
                         predicted_yield, confidence_lower, confidence_upper, std_dev,
                         soil_type, humidity)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        user_data.get('user_id'),
                        user_data['username'],
                        input_data['crop_type'],      # ML feature
                        input_data['county'],         # ML feature  
                        input_data['soil_ph'],        # ML feature
                        input_data['soil_moisture'],  # ML feature
                        input_data['fertilizer_usage'], # ML feature
                        input_data['temperature'],    # ML feature
                        input_data['rainfall'],       # ML feature
                        input_data['altitude'],       # ML feature
                        prediction_result['predicted_yield'],
                        prediction_result['confidence_lower'],
                        prediction_result['confidence_upper'],
                        prediction_result['std_dev'],
                        input_data.get('soil_type'),  # Optional field
                        input_data.get('humidity')    # Optional field
                    ))
                    
                    prediction_id = cursor.lastrowid
                    conn.commit()
                
                # Update analytics in separate transaction
                self._update_analytics_safe(user_data, input_data, prediction_result)
                
                st.success(f"✅ Prediction #{prediction_id} saved with ML features!")
                return True
                
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    st.error(f"Error storing prediction after {attempt + 1} attempts: {str(e)}")
                    return False
            except Exception as e:
                st.error(f"Error storing prediction: {str(e)}")
                return False
            finally:
                if conn:
                    conn.close()
        
        return False
    
    def _update_analytics_safe(self, user_data, input_data, prediction_result):
        """Update analytics with comprehensive ML feature tracking"""
        max_retries = 5
        retry_delay = 0.2
        
        for attempt in range(max_retries):
            conn = None
            try:
                with self._write_lock:
                    conn = self.get_connection()
                    if not conn:
                        return False
                    
                    cursor = conn.cursor()
                    
                    # Check if analytics entry exists
                    cursor.execute("""
                        SELECT id, average_yield, prediction_count, total_yield, min_yield, max_yield, yield_std_dev,
                               avg_soil_ph, avg_soil_moisture, avg_fertilizer_usage, avg_temperature, avg_rainfall
                        FROM analytics 
                        WHERE username = ? AND crop_type = ? AND county = ?
                    """, (user_data['username'], input_data['crop_type'], input_data['county']))
                    
                    existing_analytics = cursor.fetchone()
                    
                    current_yield = prediction_result['predicted_yield']
                    
                    if existing_analytics:
                        # Update existing analytics with ML features
                        (analytics_id, current_avg, current_count, current_total, 
                         current_min, current_max, current_std_dev,
                         avg_ph, avg_moisture, avg_fert, avg_temp, avg_rain) = existing_analytics
                        
                        new_count = current_count + 1
                        new_total = current_total + current_yield
                        new_avg = new_total / new_count
                        new_min = min(current_min, current_yield)
                        new_max = max(current_max, current_yield)
                        
                        # Update feature averages (weighted by count)
                        new_avg_ph = ((avg_ph * current_count) + input_data['soil_ph']) / new_count
                        new_avg_moisture = ((avg_moisture * current_count) + input_data['soil_moisture']) / new_count
                        new_avg_fert = ((avg_fert * current_count) + input_data['fertilizer_usage']) / new_count
                        new_avg_temp = ((avg_temp * current_count) + input_data['temperature']) / new_count
                        new_avg_rain = ((avg_rain * current_count) + input_data['rainfall']) / new_count
                        
                        # Calculate new standard deviation (simplified)
                        variance_numerator = (current_std_dev ** 2) * current_count + (current_yield - new_avg) * (current_yield - current_avg)
                        new_std_dev = np.sqrt(variance_numerator / new_count) if new_count > 1 else 0
                        
                        cursor.execute("""
                            UPDATE analytics 
                            SET average_yield = ?, prediction_count = ?, total_yield = ?, 
                                min_yield = ?, max_yield = ?, yield_std_dev = ?,
                                avg_soil_ph = ?, avg_soil_moisture = ?, avg_fertilizer_usage = ?,
                                avg_temperature = ?, avg_rainfall = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """, (new_avg, new_count, new_total, new_min, new_max, new_std_dev,
                              new_avg_ph, new_avg_moisture, new_avg_fert, new_avg_temp, new_avg_rain,
                              analytics_id))
                        
                    else:
                        # Insert new analytics entry with ML features
                        cursor.execute("""
                            INSERT INTO analytics 
                            (username, crop_type, county, average_yield, prediction_count, 
                             total_yield, min_yield, max_yield, yield_std_dev,
                             avg_soil_ph, avg_soil_moisture, avg_fertilizer_usage, 
                             avg_temperature, avg_rainfall)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            user_data['username'],
                            input_data['crop_type'],
                            input_data['county'],
                            current_yield,  # average_yield
                            1,              # prediction_count
                            current_yield,  # total_yield
                            current_yield,  # min_yield
                            current_yield,  # max_yield
                            0,              # yield_std_dev (will update with more data)
                            input_data['soil_ph'],
                            input_data['soil_moisture'],
                            input_data['fertilizer_usage'],
                            input_data['temperature'],
                            input_data['rainfall']
                        ))
                    
                    conn.commit()
                    return True
                    
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                    continue
                else:
                    st.warning(f"Could not update analytics after {attempt + 1} attempts: {str(e)}")
                    return False
            except Exception as e:
                st.warning(f"Error updating analytics: {str(e)}")
                return False
            finally:
                if conn:
                    conn.close()
        
        st.warning("Analytics update failed after all retries, but prediction was saved.")
        return False
    
    def store_model_performance(self, performance_metrics, model_version="v1.0"):
        """Store ML model performance metrics"""
        conn = self.get_connection()
        if conn:
            try:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO model_performance 
                    (model_version, train_r2, test_r2, train_mae, test_mae, 
                     train_rmse, test_rmse, cv_scores_mean, cv_scores_std, 
                     feature_count, training_samples)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_version,
                    performance_metrics['train_r2'],
                    performance_metrics['test_r2'],
                    performance_metrics['train_mae'],
                    performance_metrics['test_mae'],
                    performance_metrics['train_rmse'],
                    performance_metrics['test_rmse'],
                    performance_metrics['cv_scores'].mean(),
                    performance_metrics['cv_scores'].std(),
                    len(performance_metrics.get('feature_names', [])),
                    performance_metrics.get('training_samples', 0)
                ))
                
                conn.commit()
                return True
            except Exception as e:
                st.error(f"Error storing model performance: {str(e)}")
                return False
            finally:
                conn.close()
        return False
    
    def store_feature_importance(self, importance_df, model_version="v1.0"):
        """Store feature importance data"""
        conn = self.get_connection()
        if conn:
            try:
                cursor = conn.cursor()
                
                for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
                    cursor.execute("""
                        INSERT OR REPLACE INTO feature_importance 
                        (model_version, feature_name, importance, rank)
                        VALUES (?, ?, ?, ?)
                    """, (model_version, row['feature'], row['importance'], rank))
                
                conn.commit()
                return True
            except Exception as e:
                st.error(f"Error storing feature importance: {str(e)}")
                return False
            finally:
                conn.close()
        return False
    
    def get_predictions_data(self):
        """Get predictions data with ML features"""
        conn = self.get_connection()
        if conn:
            try:
                query = """
                    SELECT 
                        id,
                        username,
                        crop_type,
                        county,
                        soil_ph,
                        soil_moisture,
                        fertilizer_usage,
                        temperature,
                        rainfall,
                        altitude,
                        predicted_yield,
                        confidence_lower,
                        confidence_upper,
                        std_dev,
                        soil_type,
                        humidity,
                        model_version,
                        created_at
                    FROM predictions 
                    ORDER BY created_at DESC
                """
                df = pd.read_sql_query(query, conn)
                return df
            except Exception as e:
                st.error(f"Error fetching predictions: {str(e)}")
                return pd.DataFrame()
            finally:
                conn.close()
        return pd.DataFrame()
    
    def get_analytics_data(self):
        """Get analytics data with ML feature averages"""
        conn = self.get_connection()
        if conn:
            try:
                query = """
                    SELECT 
                        id,
                        username,
                        crop_type,
                        county,
                        average_yield,
                        prediction_count,
                        total_yield,
                        min_yield,
                        max_yield,
                        yield_std_dev,
                        avg_soil_ph,
                        avg_soil_moisture,
                        avg_fertilizer_usage,
                        avg_temperature,
                        avg_rainfall,
                        created_at,
                        updated_at
                    FROM analytics 
                    ORDER BY updated_at DESC
                """
                df = pd.read_sql_query(query, conn)
                return df
            except Exception as e:
                st.error(f"Error fetching analytics: {str(e)}")
                return pd.DataFrame()
            finally:
                conn.close()
        return pd.DataFrame()
    
    def get_users_data(self):
        """Get users data"""
        conn = self.get_connection()
        if conn:
            try:
                query = """
                    SELECT 
                        id,
                        username,
                        user_type,
                        county,
                        created_at,
                        last_login,
                        is_active
                    FROM users 
                    ORDER BY created_at DESC
                """
                df = pd.read_sql_query(query, conn)
                return df
            except Exception as e:
                return pd.DataFrame()
            finally:
                conn.close()
        return pd.DataFrame()
    
    def get_model_performance(self, model_version="v1.0"):
        """Get model performance data"""
        conn = self.get_connection()
        if conn:
            try:
                query = """
                    SELECT * FROM model_performance 
                    WHERE model_version = ?
                    ORDER BY created_at DESC 
                    LIMIT 1
                """
                df = pd.read_sql_query(query, conn, params=(model_version,))
                return df
            except Exception as e:
                return pd.DataFrame()
            finally:
                conn.close()
        return pd.DataFrame()
    
    def get_feature_importance(self, model_version="v1.0"):
        """Get feature importance data"""
        conn = self.get_connection()
        if conn:
            try:
                query = """
                    SELECT feature_name, importance, rank 
                    FROM feature_importance 
                    WHERE model_version = ?
                    ORDER BY rank ASC
                """
                df = pd.read_sql_query(query, conn, params=(model_version,))
                return df
            except Exception as e:
                return pd.DataFrame()
            finally:
                conn.close()
        return pd.DataFrame()
    
    def get_database_stats(self):
        """Get comprehensive database statistics"""
        conn = self.get_connection()
        if conn:
            try:
                cursor = conn.cursor()
                
                # Table counts
                cursor.execute("SELECT COUNT(*) FROM users")
                users_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM predictions")
                predictions_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM analytics")
                analytics_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM model_performance")
                model_perf_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM feature_importance")
                feature_imp_count = cursor.fetchone()[0]
                
                # Prediction statistics
                cursor.execute("SELECT AVG(predicted_yield), MIN(predicted_yield), MAX(predicted_yield) FROM predictions")
                yield_stats = cursor.fetchone()
                
                # User statistics
                cursor.execute("SELECT COUNT(DISTINCT username) FROM predictions")
                active_users = cursor.fetchone()[0]
                
                return {
                    'users_count': users_count,
                    'predictions_count': predictions_count,
                    'analytics_count': analytics_count,
                    'model_perf_count': model_perf_count,
                    'feature_imp_count': feature_imp_count,
                    'avg_yield': round(yield_stats[0], 1) if yield_stats[0] else 0,
                    'min_yield': round(yield_stats[1], 1) if yield_stats[1] else 0,
                    'max_yield': round(yield_stats[2], 1) if yield_stats[2] else 0,
                    'active_users': active_users
                }
            except Exception as e:
                return {'error': str(e)}
            finally:
                conn.close()
        return {'error': 'No connection'}
    
    def debug_database(self):
        """Debug method to check database structure"""
        conn = self.get_connection()
        if conn:
            try:
                cursor = conn.cursor()
                
                st.info("🔍 DATABASE STRUCTURE")
                
                tables = ['users', 'predictions', 'analytics', 'model_performance', 'feature_importance']
                for table in tables:
                    st.info(f"📊 {table.upper()} table columns:")
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()
                    for col in columns:
                        st.write(f"  {col[1]} ({col[2]}) - {'PK' if col[5] else ''}")
                
                # Show current data counts
                st.info("📈 CURRENT DATA COUNTS:")
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    st.write(f"  {table}: {count} records")
                
                # Show sample predictions
                st.info("🌾 SAMPLE PREDICTIONS:")
                cursor.execute("SELECT crop_type, county, predicted_yield, created_at FROM predictions LIMIT 5")
                sample_predictions = cursor.fetchall()
                if sample_predictions:
                    for record in sample_predictions:
                        st.write(f"  Crop: {record[0]}, County: {record[1]}, Yield: {record[2]:.0f} kg/ha, Date: {record[3]}")
                else:
                    st.write("  No prediction records found")
                    
            except Exception as e:
                st.error(f"Debug error: {e}")
            finally:
                conn.close()
    
    def debug_analytics(self):
        """Debug method to check analytics data"""
        conn = self.get_connection()
        if conn:
            try:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM analytics")
                count = cursor.fetchone()[0]
                st.info(f"Total analytics records: {count}")
                
                if count > 0:
                    cursor.execute("""
                        SELECT username, crop_type, county, average_yield, prediction_count 
                        FROM analytics 
                        ORDER BY prediction_count DESC 
                        LIMIT 10
                    """)
                    records = cursor.fetchall()
                    st.info("Top Analytics Records:")
                    for record in records:
                        st.write(f"  User: {record[0]}, Crop: {record[1]}, County: {record[2]}, Avg Yield: {record[3]:.0f} kg/ha, Count: {record[4]}")
                else:
                    st.warning("No analytics records found")
                    
            except Exception as e:
                st.error(f"Debug error: {e}")
            finally:
                conn.close()
    
    def reset_database(self):
        """Reset database for testing"""
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
                st.success("🗑️ Old database removed")
            
            # Reinitialize tables
            self._initialize_tables()
            st.success("🆕 New database created successfully!")
            return True
        except Exception as e:
            st.error(f"Error resetting database: {e}")
            return False
    
    def get_database_info(self):
        """Get database file information"""
        try:
            if os.path.exists(self.db_path):
                size = os.path.getsize(self.db_path)
                # Get table counts
                stats = self.get_database_stats()
                
                return {
                    'path': self.db_path,
                    'size_bytes': size,
                    'size_mb': round(size / (1024 * 1024), 2),
                    'exists': True,
                    'table_counts': stats
                }
            else:
                return {
                    'path': self.db_path,
                    'exists': False
                }
        except Exception as e:
            return {'error': str(e)}
    
    def export_predictions_csv(self):
        """Export predictions data as CSV"""
        df = self.get_predictions_data()
        if not df.empty:
            return df.to_csv(index=False)
        return None
    
    def export_analytics_csv(self):
        """Export analytics data as CSV"""
        df = self.get_analytics_data()
        if not df.empty:
            return df.to_csv(index=False)
        return None
    
    def get_yield_trends(self, days=30):
        """Get yield trends for the last N days"""
        conn = self.get_connection()
        if conn:
            try:
                query = """
                    SELECT 
                        DATE(created_at) as date,
                        AVG(predicted_yield) as avg_yield,
                        COUNT(*) as prediction_count
                    FROM predictions 
                    WHERE created_at >= DATE('now', ?)
                    GROUP BY DATE(created_at)
                    ORDER BY date DESC
                """
                df = pd.read_sql_query(query, conn, params=(f'-{days} days',))
                return df
            except Exception as e:
                return pd.DataFrame()
            finally:
                conn.close()
        return pd.DataFrame()