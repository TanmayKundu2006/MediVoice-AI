import speech_recognition as sr
import pyttsx3
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

class CSVMedicalDiagnosis:
    def __init__(self, csv_file_path='test_data.csv'):
        print("Loading medical dataset from CSV...")
        
        # Load models
        self.nlp = spacy.load("en_core_web_sm")
        self.tts_engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        
        # Load and process CSV data
        self.medical_df = pd.read_csv(csv_file_path)
        self.process_dataset()
        
        print(f"✅ Loaded {len(self.diseases)} diseases with {len(self.symptoms_list)} symptoms")
        print(f"📊 Dataset shape: {self.medical_df.shape}")
    
    def process_dataset(self):
        """Process the CSV dataset into usable format"""
        # Extract symptoms (all columns except the last one 'prognosis')
        self.symptoms_list = self.medical_df.columns[:-1].tolist()
        
        # Extract diseases
        self.diseases = self.medical_df['prognosis'].unique().tolist()
        
        # Create disease-to-symptoms mapping
        self.disease_symptoms = {}
        self.symptom_frequencies = {}
        
        for _, row in self.medical_df.iterrows():
            disease = row['prognosis']
            symptoms_present = []
            
            for symptom in self.symptoms_list:
                if row[symptom] == 1:
                    symptoms_present.append(symptom)
            
            self.disease_symptoms[disease] = symptoms_present
        
        # Create human-readable symptom mappings
        self.symptom_mapping = self.create_symptom_mapping()
    
    def create_symptom_mapping(self):
        """Create mapping from spoken words to dataset symptom names"""
        mapping = {
            # Pain-related symptoms
            'pain': 'joint_pain', 'stomach pain': 'stomach_pain', 'headache': 'headache',
            'chest pain': 'chest_pain', 'muscle pain': 'muscle_pain', 'back pain': 'back_pain',
            'neck pain': 'neck_pain', 'knee pain': 'knee_pain', 'hip pain': 'hip_joint_pain',
            'anal pain': 'pain_in_anal_region',
            
            # Respiratory symptoms
            'cough': 'cough', 'sneezing': 'continuous_sneezing', 'breathlessness': 'breathlessness',
            'shortness of breath': 'breathlessness', 'runny nose': 'runny_nose', 
            'congestion': 'congestion', 'phlegm': 'phlegm',
            
            # Fever and temperature
            'fever': 'high_fever', 'mild fever': 'mild_fever', 'chills': 'chills',
            'shivering': 'shivering', 'sweating': 'sweating',
            
            # Digestive symptoms
            'vomiting': 'vomiting', 'nausea': 'nausea', 'diarrhea': 'diarrhoea',
            'constipation': 'constipation', 'indigestion': 'indigestion', 
            'acidity': 'acidity', 'loss of appetite': 'loss_of_appetite',
            'abdominal pain': 'abdominal_pain', 'stomach ache': 'stomach_pain',
            
            # Skin symptoms
            'itching': 'itching', 'rash': 'skin_rash', 'skin eruptions': 'nodal_skin_eruptions',
            
            # General symptoms
            'fatigue': 'fatigue', 'weakness': 'weakness_in_limbs', 'lethargy': 'lethargy',
            'weight loss': 'weight_loss', 'weight gain': 'weight_gain', 'dizziness': 'dizziness',
            'restlessness': 'restlessness', 'anxiety': 'anxiety',
            
            # Urinary symptoms
            'burning urination': 'burning_micturition', 'dark urine': 'dark_urine',
            'yellow urine': 'yellow_urine', 'spotting urination': 'spotting_ urination',
            
            # Other specific symptoms
            'yellow skin': 'yellowish_skin', 'yellow eyes': 'yellowing_of_eyes',
            'sunken eyes': 'sunken_eyes', 'blurred vision': 'blurred_and_distorted_vision',
            'red eyes': 'redness_of_eyes', 'throat irritation': 'throat_irritation',
            'sinus pressure': 'sinus_pressure'
        }
        
        # Also add direct mappings for symptoms that match closely
        for symptom in self.symptoms_list:
            human_readable = symptom.replace('_', ' ').replace('  ', ' ')
            mapping[human_readable] = symptom
        
        return mapping
    
    def extract_symptoms_from_speech(self, text):
        """Extract symptoms from spoken text and map to dataset columns"""
        text_lower = text.lower()
        found_symptoms = []
        
        # First check multi-word symptoms
        for human_symptom, dataset_symptom in self.symptom_mapping.items():
            if ' ' in human_symptom and human_symptom in text_lower:
                if dataset_symptom in self.symptoms_list:
                    found_symptoms.append(dataset_symptom)
        
        # Then check single-word symptoms
        words = text_lower.split()
        for word in words:
            if word in self.symptom_mapping:
                dataset_symptom = self.symptom_mapping[word]
                if dataset_symptom in self.symptoms_list and dataset_symptom not in found_symptoms:
                    found_symptoms.append(dataset_symptom)
        
        # Also check for partial matches
        for symptom in self.symptoms_list:
            human_readable = symptom.replace('_', ' ')
            if human_readable in text_lower and symptom not in found_symptoms:
                found_symptoms.append(symptom)
        
        return list(set(found_symptoms))
    
    def calculate_disease_probability(self, user_symptoms):
        """Calculate probability scores for each disease based on symptom matches"""
        disease_scores = {}
        
        for disease, typical_symptoms in self.disease_symptoms.items():
            if not typical_symptoms:  # Skip diseases with no symptoms
                continue
                
            # Find matching symptoms
            matches = set(user_symptoms) & set(typical_symptoms)
            
            if matches:
                # Calculate match percentage
                match_percentage = (len(matches) / len(typical_symptoms)) * 100
                
                # Calculate score based on match percentage and number of matches
                base_score = len(matches)  # One point per match
                weighted_score = match_percentage / 10  # Percentage factor
                
                total_score = base_score + weighted_score
                
                disease_scores[disease] = {
                    'score': total_score,
                    'match_percentage': match_percentage,
                    'matched_symptoms': list(matches),
                    'total_typical_symptoms': len(typical_symptoms),
                    'matches_count': len(matches)
                }
        
        return disease_scores
    
    def get_top_diagnoses(self, user_symptoms, top_n=5):
        """Get top N most likely diagnoses"""
        if not user_symptoms:
            return None
        
        disease_scores = self.calculate_disease_probability(user_symptoms)
        
        if not disease_scores:
            return None
        
        # Sort by score (descending)
        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        return sorted_diseases[:top_n]
    
    def symptom_similarity_match(self, user_symptoms):
        """Advanced matching using symptom similarity"""
        # Create symptom vectors for each disease
        disease_vectors = {}
        
        for disease, symptoms in self.disease_symptoms.items():
            # Create a binary vector for this disease
            vector = [1 if symptom in symptoms else 0 for symptom in self.symptoms_list]
            disease_vectors[disease] = vector
        
        # Create user symptom vector
        user_vector = [1 if symptom in user_symptoms else 0 for symptom in self.symptoms_list]
        
        # Calculate cosine similarity
        similarities = {}
        for disease, d_vector in disease_vectors.items():
            if sum(d_vector) > 0:  # Only consider diseases with symptoms
                similarity = cosine_similarity([user_vector], [d_vector])[0][0]
                similarities[disease] = similarity
        
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    def speak(self, text):
        """Text to speech"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def analyze_speech_symptoms(self):
        """Main function to analyze symptoms from speech"""
        try:
            
            
            
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("🔊 Listening... (speak now)")
                
                # Listen for audio with timeout
                audio = self.recognizer.listen(source, timeout=15)
                print("✅ Processing your speech...")
                
                # Convert speech to text
                spoken_text = self.recognizer.recognize_google(audio)
                spoken_text_lower = spoken_text.lower()
                
                print(f" You said: '{spoken_text}'")
                self.speak(f"I heard: {spoken_text}")
                
                # Extract symptoms from speech
                detected_symptoms = self.extract_symptoms_from_speech(spoken_text_lower)
                
                if detected_symptoms:
                    print(f" Detected {len(detected_symptoms)} symptoms: {', '.join(detected_symptoms)}")
                    
                    # Get human-readable symptom names
                    human_symptoms = [symptom.replace('_', ' ') for symptom in detected_symptoms]
                    print(f" Symptoms: {', '.join(human_symptoms)}")
                    
                    # Get diagnoses using basic matching
                    basic_results = self.get_top_diagnoses(detected_symptoms, top_n=5)
                    
                    # Get diagnoses using advanced similarity matching
                    similarity_results = self.symptom_similarity_match(detected_symptoms)
                    
                    # Display results
                    self.display_results(detected_symptoms, basic_results, similarity_results, spoken_text)
                    
                else:
                    print("❌ No recognizable symptoms detected.")
                    print("💡 Try using specific symptom names from the list above.")
                    self.speak("I didn't detect any specific symptoms. Please mention symptoms like fever, cough, or headache.")
                    
        except sr.WaitTimeoutError:
            print("⏰ No speech detected within 15 seconds.")
            self.speak("I didn't hear anything. Please try again.")
        except sr.UnknownValueError:
            print("🔇 Could not understand the audio. Please speak clearly.")
            self.speak("I couldn't understand what you said. Please speak clearly.")
        except Exception as e:
            print(f"💥 An error occurred: {e}")
            self.speak("An error occurred during analysis.")
    
    def display_results(self, symptoms, basic_results, similarity_results, original_text):
        """Display comprehensive results"""
        print("\n" + "="*80)
        print("🏥 MEDICAL SYMPTOM ANALYSIS RESULTS")
        print("="*80)
        print("⚠️  DISCLAIMER: This is for educational purposes only.")
        print("   Always consult healthcare professionals for medical diagnosis.")
        print("-"*80)
        
        print(f"📊 Analysis based on {len(symptoms)} detected symptoms")
        print(f"💬 Original statement: '{original_text}'")
        
        if basic_results:
            print(f"\n🎯 TOP 5 DIAGNOSES (Basic Matching):")
            print("-"*50)
            
            for i, (disease, data) in enumerate(basic_results, 1):
                print(f"{i}. {disease}")
                print(f"   📈 Match Score: {data['score']:.2f}")
                print(f"   📊 Match Percentage: {data['match_percentage']:.1f}%")
                print(f"   ✅ Matched {data['matches_count']}/{data['total_typical_symptoms']} symptoms")
                
                # Show matched symptoms
                human_matches = [s.replace('_', ' ') for s in data['matched_symptoms'][:5]]
                print(f"   🔍 Symptoms: {', '.join(human_matches)}")
                print()
        
        if similarity_results:
            print(f"\n🔬 TOP 5 DIAGNOSES (Similarity Analysis):")
            print("-"*50)
            
            for i, (disease, similarity) in enumerate(similarity_results[:5], 1):
                typical_symptoms = self.disease_symptoms.get(disease, [])
                human_symptoms = [s.replace('_', ' ') for s in typical_symptoms[:3]]
                
                print(f"{i}. {disease}")
                print(f"   📈 Similarity Score: {similarity:.3f}")
                print(f"   📋 Typical symptoms: {', '.join(human_symptoms)}...")
                print()
        
        # Provide voice feedback
        if basic_results:
            best_match = basic_results[0][0]
            match_data = basic_results[0][1]
            
            voice_text = (
                f"Based on your symptoms, the most likely condition is {best_match}. "
                f"Matched {match_data['matches_count']} out of {match_data['total_typical_symptoms']} typical symptoms. "
                f"Please consult a doctor for proper diagnosis."
            )
        else:
            voice_text = "No strong matches found. Please consult a healthcare professional."
        
        self.speak(voice_text)
        
        # Show available symptoms for reference
        print(f"\n💡 Tip: You mentioned {len(symptoms)} symptoms. The dataset contains {len(self.symptoms_list)} possible symptoms.")
        print("   For better accuracy, try to be more specific about your symptoms.")

# Additional utility functions
def show_available_symptoms(analyzer):
    """Display available symptoms from the dataset"""
    print(f"\n📋 AVAILABLE SYMPTOMS ({len(analyzer.symptoms_list)} total):")
    print("-"*60)
    
    # Group symptoms by category for better readability
    pain_symptoms = [s for s in analyzer.symptoms_list if 'pain' in s]
    fever_symptoms = [s for s in analyzer.symptoms_list if 'fever' in s or 'chills' in s]
    digestive_symptoms = [s for s in analyzer.symptoms_list if any(word in s for word in ['vomiting', 'nausea', 'diarrhoea', 'constipation'])]
    skin_symptoms = [s for s in analyzer.symptoms_list if any(word in s for word in ['itching', 'rash', 'skin'])]
    
    print("🌡️  Fever/Temperature: " + ", ".join([s.replace('_', ' ') for s in fever_symptoms[:5]]) + "...")
    print("😖 Pain: " + ", ".join([s.replace('_', ' ') for s in pain_symptoms[:5]]) + "...")
    print("🤢 Digestive: " + ", ".join([s.replace('_', ' ') for s in digestive_symptoms[:5]]) + "...")
    print("🧴 Skin: " + ", ".join([s.replace('_', ' ') for s in skin_symptoms[:5]]) + "...")
    print(f"📊 And {len(analyzer.symptoms_list) - 20} more symptoms...")

def show_available_diseases(analyzer):
    """Display available diseases from the dataset"""
    print(f"\n🏥 AVAILABLE DISEASES ({len(analyzer.diseases)} total):")
    print("-"*60)
    
    for i, disease in enumerate(analyzer.diseases[:10], 1):
        typical_symptoms = analyzer.disease_symptoms.get(disease, [])
        print(f"{i}. {disease} ({len(typical_symptoms)} typical symptoms)")
    
    if len(analyzer.diseases) > 10:
        print(f"... and {len(analyzer.diseases) - 10} more diseases")

# Main execution
if __name__ == "__main__":
    print("🏥 CSV-BASED MEDICAL SPEECH DIAGNOSIS SYSTEM")
    print("="*60)
    
    try:
        # Initialize the system
        medical_analyzer = CSVMedicalDiagnosis('test_data.csv')
        
        # Show dataset information
        show_available_symptoms(medical_analyzer)
        show_available_diseases(medical_analyzer)
        
        print("\n" + "="*60)
        print("🎤 READY FOR SPEECH INPUT")
        print("="*60)
        
        # Run the analysis
        medical_analyzer.analyze_speech_symptoms()
        
    except FileNotFoundError:
        print("❌ Error: CSV file 'test_data.csv' not found.")
        print("💡 Please make sure the file is in the same directory as this script.")
    except Exception as e:
        print(f"💥 Error initializing system: {e}")