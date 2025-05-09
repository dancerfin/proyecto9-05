from __future__ import division
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.pipeline import make_pipeline
import joblib
from collections import deque
import time
import statistics

class HybridDDoSDetector:
    def __init__(self):
        """
        Modelo híbrido mejorado que combina Random Forest y SVM para detección de DDoS.
        
        Mejoras implementadas:
        - Balanceo de clases con class_weight
        - Normalización de características
        - Umbrales adaptativos
        - Sistema de votación temporal
        - Votación ponderada mejorada
        - Mecanismo de fallback robusto
        """
        self.rf_model = None
        self.svm_model = None
        self.scaler = None
        self.last_predictions = deque(maxlen=20)  # Para seguimiento temporal
        self.prediction_history = deque(maxlen=100)  # Historial largo para umbrales
        self.attack_prob_history = deque(maxlen=100)  # Historial de probabilidades
        self.dynamic_threshold = 0.7  # Umbral inicial
        self.last_update_time = time.time()
        
        # Configuración de modelos
        self.rf_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 2,
            'class_weight': 'balanced_subsample',
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.svm_params = {
            'kernel': 'rbf',
            'C': 1.5,
            'gamma': 'scale',
            'probability': True,
            'class_weight': 'balanced'
        }
        
        try:
            # Intentar cargar modelos pre-entrenados
            self.load_models()
            print("Modelos híbridos cargados exitosamente")
        except Exception as e:
            print(f"Error al cargar modelos: {str(e)}. Entrenando nuevos modelos...")
            self.train_hybrid_model()
    
    def load_models(self):
        """Carga los modelos y el scaler desde archivos"""
        self.rf_model = joblib.load('rf_model.pkl')
        self.svm_model = joblib.load('svm_model.pkl')
        self.scaler = joblib.load('scaler.pkl')
    
    def save_models(self):
        """Guarda los modelos entrenados y el scaler"""
        joblib.dump(self.rf_model, 'rf_model.pkl')
        joblib.dump(self.svm_model, 'svm_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
    
    def calculate_class_weights(self, y):
        """Calcula pesos de clases para manejar desbalance"""
        classes = np.unique(y)
        weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    
    def train_hybrid_model(self):
        """Entrena ambos modelos con los datos de result.csv"""
        # Cargar y preparar datos
        data = np.loadtxt(open('result.csv', 'rb'), delimiter=',', dtype='str')
        X = data[:, 0:5].astype(float)
        y = data[:, 5]
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Normalización de características
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Calcular pesos de clases
        class_weights = self.calculate_class_weights(y_train)
        self.rf_params['class_weight'] = class_weights
        self.svm_params['class_weight'] = class_weights
        
        # Entrenar Random Forest con pipeline
        self.rf_model = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(**self.rf_params)
        )
        self.rf_model.fit(X_train, y_train)
        
        # Entrenar SVM con pipeline
        self.svm_model = make_pipeline(
            StandardScaler(),
            svm.SVC(**self.svm_params)
        )
        self.svm_model.fit(X_train, y_train)
        
        # Evaluar modelos
        print("\nEvaluación Random Forest:")
        y_pred_rf = self.rf_model.predict(X_test)
        print(classification_report(y_test, y_pred_rf))
        
        print("\nEvaluación SVM:")
        y_pred_svm = self.svm_model.predict(X_test)
        print(classification_report(y_test, y_pred_svm))
        
        # Calcular umbral inicial basado en los datos de entrenamiento
        self._calculate_initial_threshold(X_train, y_train)
        
        # Guardar modelos
        self.save_models()
    
    def _calculate_initial_threshold(self, X, y):
        """Calcula umbral inicial basado en probabilidades de entrenamiento"""
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        svm_proba = self.svm_model.predict_proba(X)[:, 1]
        combined_proba = (rf_proba + svm_proba) / 2
        self.dynamic_threshold = np.percentile(combined_proba[y == '1'], 25)  # Percentil 25 de ataques
    
    def _update_dynamic_threshold(self):
        """Actualiza el umbral dinámico basado en el historial"""
        if len(self.attack_prob_history) > 20:
            recent_probs = list(self.attack_prob_history)[-20:]
            mean_prob = statistics.mean(recent_probs)
            std_prob = statistics.stdev(recent_probs)
            
            # Ajustar umbral basado en la media y desviación estándar
            self.dynamic_threshold = max(0.5, min(0.9, mean_prob + std_prob * 0.5))
            
            # Actualizar solo cada 5 minutos
            current_time = time.time()
            if current_time - self.last_update_time > 300:
                self.last_update_time = current_time
                print(f"Umbral dinámico actualizado: {self.dynamic_threshold:.2f}")
    
    def hybrid_predict(self, features):
        """
        Predicción híbrida mejorada que combina ambos modelos con:
        - Normalización de características
        - Umbrales dinámicos
        - Votación ponderada
        - Mecanismo de fallback
        """
        try:
            # Convertir a array numpy y normalizar
            features = np.array(features).reshape(1, -1).astype(float)
            if self.scaler:
                features = self.scaler.transform(features)
            
            # Obtener probabilidades
            rf_proba = self.rf_model.predict_proba(features)[0]
            svm_proba = self.svm_model.predict_proba(features)[0]
            
            # Obtener clases y confianzas
            rf_class = self.rf_model.classes_[np.argmax(rf_proba)]
            rf_confidence = np.max(rf_proba)
            rf_attack_prob = rf_proba[1] if '1' in self.rf_model.classes_ else 0
            
            svm_class = self.svm_model.classes_[np.argmax(svm_proba)]
            svm_confidence = np.max(svm_proba)
            svm_attack_prob = svm_proba[1] if '1' in self.svm_model.classes_ else 0
            
            # Calcular probabilidad combinada ponderada
            combined_prob = (rf_attack_prob * 0.6 + svm_attack_prob * 0.4)
            self.attack_prob_history.append(combined_prob)
            
            # Actualizar umbral dinámico
            self._update_dynamic_threshold()
            
            # Motor de decisión mejorado
            final_pred = self._improved_decision_engine(
                rf_class, rf_confidence, rf_attack_prob,
                svm_class, svm_confidence, svm_attack_prob,
                combined_prob
            )
            
            # Guardar predicción para seguimiento temporal
            self.last_predictions.append(final_pred)
            self.prediction_history.append(final_pred)
            
            # Sistema de votación temporal
            if len(self.last_predictions) >= 10:
                recent_attacks = list(self.last_predictions).count('1')
                if recent_attacks >= 6:  # 60% de las últimas predicciones son ataques
                    final_pred = '1'
                elif recent_attacks <= 2:  # Menos del 20% son ataques
                    final_pred = '0'
            
            return [final_pred]
            
        except Exception as e:
            print(f"Error en predicción híbrida: {str(e)}")
            # Mecanismo de fallback: si hay historial, usar la moda
            if len(self.prediction_history) > 0:
                fallback = statistics.mode(self.prediction_history)
                return [fallback]
            return ['0']  # Por defecto asume tráfico normal
    
    def _improved_decision_engine(self, rf_class, rf_conf, rf_attack, 
                                svm_class, svm_conf, svm_attack, combined_prob):
        """
        Motor de decisión mejorado:
        1. Si ambos modelos están muy seguros y coinciden -> usa ese resultado
        2. Si un modelo está mucho más seguro que el otro -> usa ese
        3. Si ambos están inseguros -> prioriza RF pero con umbral bajo
        4. Si hay discordancia pero alta confianza -> prioriza detección de ataques
        """
        # Definir umbrales
        high_conf = 0.85
        low_conf = 0.6
        conf_diff = 0.2
        
        # Caso 1: Ambos seguros y coinciden
        if (rf_conf > high_conf and svm_conf > high_conf and 
            rf_class == svm_class):
            return rf_class
        
        # Caso 2: Un modelo mucho más seguro que el otro
        if rf_conf > svm_conf + conf_diff:
            return rf_class
        elif svm_conf > rf_conf + conf_diff:
            return svm_class
        
        # Caso 3: Ambos inseguros
        if rf_conf < low_conf and svm_conf < low_conf:
            # Priorizar RF pero con umbral bajo
            return '1' if combined_prob > self.dynamic_threshold * 0.8 else '0'
        
        # Caso 4: Discordancia pero alta confianza en al menos uno
        if rf_conf > high_conf or svm_conf > high_conf:
            # Priorizar detección de ataques
            if rf_class == '1' or svm_class == '1':
                return '1'
            return '0'
        
        # Caso por defecto: usar probabilidad combinada con umbral dinámico
        return '1' if combined_prob > self.dynamic_threshold else '0'

class MachineLearningAlgo:
    """
    Wrapper para mantener compatibilidad con el código existente
    mientras usamos el nuevo detector híbrido mejorado
    """
    def __init__(self):
        self.detector = HybridDDoSDetector()
    
    def classify(self, data):
        return self.detector.hybrid_predict(data)