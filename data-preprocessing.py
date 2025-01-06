import anthropic
import json
import os
from typing import List, Dict, Any, Optional

class CharacterBehaviorDataPreprocessor:
    """
    Advanced data preprocessing pipeline for character behavior modeling
    
    Leverages Claude.ai Pro API for intelligent feature augmentation
    """
    
    def __init__(
        self, 
        novel_text: str,
        api_key: Optional[str] = None,
        novel_title: str = "Anne of Green Gables",
        author: str = "Lucy Maud Montgomery"
    ):
        # Core text processing configuration
        self.novel_text = novel_text
        self.novel_title = novel_title
        self.author = author
        
        # Claude.ai API Client
        self.claude_client = anthropic.Anthropic(api_key=api_key) if api_key else None
        
        # Initialize NLP processing tools
        import spacy
        import nltk
        nltk.download('punkt', quiet=True)
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """
        Initialize tokenization method
        Supports fallback mechanisms
        """
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained("facebook/opt-350m")
        except ImportError:
            # Fallback tokenization
            import nltk
            return nltk.word_tokenize
    
    def augment_scene_features(
        self, 
        scene_context: str
    ) -> Dict[str, Any]:
        """
        Use Claude.ai Pro API for comprehensive feature augmentation
        
        Args:
            scene_context (str): Full text of the scene
        
        Returns:
            Dict: Augmented scene features with scholarly insights
        """
        if not self.claude_client:
            raise ValueError("Claude.ai API client not configured")
        
        try:
            # Construct detailed augmentation prompt
            response = self.claude_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                        Perform an advanced, multi-dimensional feature extraction for a scene 
                        from '{self.novel_title}' by {self.author}.

                        Scene Context:
                        ```
                        {scene_context}
                        ```

                        Generate a comprehensive JSON with deep insights across these dimensions:

                        1. Time-Space Coordinates
                           - Geographical context
                           - Historical period details
                           - Social environment nuances

                        2. Character History
                           - Relationship dynamics
                           - Social network mapping
                           - Character development insights

                        3. Emotional Expression
                           - Emotional landscape
                           - Communication subtexts
                           - Psychological undertones

                        4. Inner Psychological State
                           - Motivational drivers
                           - Underlying psychological mechanisms
                           - Implicit character thoughts

                        5. Conversation Dynamics
                           - Dialogue structure
                           - Power dynamics
                           - Communication strategies

                        6. Motivational Dimensions
                           - Core psychological motivations
                           - Social and personal drivers
                           - Contextual motivation analysis

                        Provide scholarly, interdisciplinary perspectives. 
                        Include literary, sociological, and psychological insights.
                        Use academic rigor and nuanced interpretation.
                        """
                    }
                ]
            )
            
            # Extract and parse the response
            feature_json = self._parse_claude_response(
                response.content[0].text
            )
            
            return feature_json
        
        except Exception as e:
            print(f"Feature augmentation error: {e}")
            # Fallback to previous feature extraction method
            return self._fallback_feature_extraction(scene_context)
    
    def _parse_claude_response(self, response_text: str) -> Dict[str, Any]:
        """
        Robust parsing of Claude.ai generated feature JSON
        
        Args:
            response_text (str): Raw response from Claude
        
        Returns:
            Dict: Parsed feature JSON
        """
        # Multiple parsing strategies
        parsing_strategies = [
            lambda x: json.loads(x),  # Standard JSON parsing
            lambda x: json.loads(x.split('```json')[-1].split('```')[0]),  # Markdown code block
            self._fallback_json_parsing  # Custom fallback parsing
        ]
        
        for strategy in parsing_strategies:
            try:
                return strategy(response_text)
            except Exception:
                continue
        
        # Ultimate fallback
        return self._fallback_feature_extraction(response_text)
    
    def _fallback_json_parsing(self, text: str) -> Dict[str, Any]:
        """
        Advanced fallback JSON parsing with error correction
        
        Args:
            text (str): Potentially malformed JSON text
        
        Returns:
            Dict: Parsed feature dictionary
        """
        import re
        
        # Remove potential markdown code block markers
        clean_text = re.sub(r'```(json)?', '', text)
        
        # Attempt more lenient parsing
        import ast
        try:
            # Try literal evaluation
            return ast.literal_eval(clean_text)
        except Exception:
            # Last resort parsing
            return self._extract_json_like_structure(clean_text)
    
    def _extract_json_like_structure(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON-like structure from text
        
        Fallback method for extreme parsing scenarios
        """
        return {
            "time_space": {"description": "Unable to fully parse scene context"},
            "character_history": {"characters": []},
            "expression": {"emotional_state": "parsing_error"},
            "inner_thought": {"psychological_insights": []},
            "conversation": {"dialogue_structure": "parsing_failed"},
            "motivation": {"core_drivers": []}
        }
    
    def _fallback_feature_extraction(self, scene_context: str) -> Dict[str, Any]:
        """
        Basic feature extraction when advanced methods fail
        
        Args:
            scene_context (str): Scene text
        
        Returns:
            Dict: Minimal feature representation
        """
        # Use SpaCy for basic feature extraction
        doc = self.nlp(scene_context)
        
        return {
            "time_space": {
                "locations": [ent.text for ent in doc.ents if ent.label_ == "GPE"],
                "temporal_markers": [token.text for token in doc if token.pos_ == "TIME"]
            },
            "character_history": {
                "characters": [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            },
            "expression": {
                "emotional_keywords": [
                    token.text for token in doc 
                    if token.text.lower() in [
                        "happy", "sad", "angry", "excited", 
                        "frustrated", "worried"
                    ]
                ]
            },
            "inner_thought": {
                "reflection_markers": [
                    sent.text for sent in doc.sents 
                    if any(word.lower() in ["thought", "wondered", "imagined"] for word in sent)
                ]
            },
            "conversation": {
                "dialogue_instances": [
                    sent.text for sent in doc.sents 
                    if '"' in sent.text
                ]
            },
            "motivation": {
                "goal_keywords": [
                    token.text for token in doc 
                    if token.text.lower() in [
                        "want", "need", "desire", "hope", 
                        "dream", "wish"
                    ]
                ]
            }
        }
    
    def process_novel(self) -> List[Dict]:
        """
        Comprehensive novel preprocessing pipeline
        
        Leverages Claude.ai Pro for advanced feature augmentation
        
        Returns:
            List of scene tokens with comprehensive features
        """
        # Scene segmentation using hybrid approach
        scenes = self.split_into_scenes()
        
        processed_scenes = []
        
        for scene in scenes:
            try:
                # Use Claude.ai for comprehensive feature augmentation
                augmented_features = self.augment_scene_features(scene)
                processed_scenes.append(augmented_features)
            
            except Exception as e:
                print(f"Scene processing error: {e}")
                # Fallback to basic feature extraction
                fallback_features = self._fallback_feature_extraction(scene)
                processed_scenes.append(fallback_features)
        
        return processed_scenes
    
    def save_processed_data(
        self, 
        output_path: str = "anne_of_green_gables_processed.json"
    ):
        """
        Save processed novel data to JSON
        
        Comprehensive error handling and logging
        """
        try:
            processed_data = self.process_novel()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2)
            
            print(f"Processed novel data saved to {output_path}")
            print(f"Total scenes processed: {len(processed_data)}")
        
        except Exception as e:
            print(f"Error saving processed data: {e}")
    
    @classmethod
    def create_preprocessing_pipeline(
        cls,
        novel_path: str,
        claude_api_key: Optional[str] = None
    ) -> 'CharacterBehaviorDataPreprocessor':
        """
        Factory method to create preprocessing pipeline
        
        Args:
            novel_path (str): Path to novel text file
            claude_api_key (Optional[str]): Claude.ai API key
        
        Returns:
            Configured preprocessing instance
        """
        # Read novel text
        with open(novel_path, 'r', encoding='utf-8') as f:
            novel_text = f.read()
        
        # Create and return preprocessor
        return cls(
            novel_text,
            api_key=claude_api_key
        )

# Example usage
def main():
    # Configuration
    NOVEL_PATH = "anne_of_green_gables.txt"
    OUTPUT_PATH = "processed_anne_of_green_gables.json"
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    
    # Create and execute preprocessing pipeline
    preprocessor = CharacterBehaviorDataPreprocessor.create_preprocessing_pipeline(
        novel_path=NOVEL_PATH,
        claude_api_key=CLAUDE_API_KEY
    )
    
    # Process and save novel data
    preprocessor.save_processed_data(OUTPUT_PATH)

if __name__ == "__main__":
    main()
    def process_novel(self) -> List[Dict]:
        """
        Comprehensive novel preprocessing pipeline with metadata-enhanced feature augmentation
        
        Core Processing Stages:
        1. Scene Segmentation
        2. Initial Feature Extraction
        3. Scholarly Metadata Enrichment
        
        Returns:
        - List of scene tokens with comprehensive, augmented features
        """
        # Scene segmentation using hybrid approach
        scenes = self.split_into_scenes()
        
        # Processed scenes with feature extraction and augmentation
        processed_scenes = []
        
        # Initialize metadata enhancer
        metadata_enhancer = MetadataEnhancedFeatureAugmenter(
            anthropic_client=self.anthropic_client,
            novel_title=self.novel_title,
            author=self.author
        )
        
        for scene in scenes:
            try:
                # Extract initial features using expert-specific methods
                initial_features = asdict(self.extract_scene_features(scene))
                
                # AI-assisted feature augmentation
                augmented_features = self.augment_scene_features(
                    initial_features, 
                    scene
                )
                
                # Metadata-enhanced scholarly enrichment
                scholarly_enriched_features = metadata_enhancer.enhance_scene_features(
                    augmented_features, 
                    scene
                )
                
                processed_scenes.append(scholarly_enriched_features)
            
            except Exception as e:
                # Robust error handling
                print(f"Error processing scene: {e}")
                # Optionally log or handle the error more comprehensively
                continue
        
        return processed_scenes
    
    def save_processed_data(
        self, 
        output_path: str = "anne_of_green_gables_processed.json"
    ):
        """
        Save processed novel data to JSON with comprehensive feature set
        
        Args:
            output_path (str): Destination file path for processed data
        """
        # Process the entire novel
        processed_data = self.process_novel()
        
        # Prepare for serialization
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2)
            
            print(f"Processed novel data saved successfully to {output_path}")
            
            # Additional metadata logging
            print(f"Total scenes processed: {len(processed_data)}")
            print("Feature dimensions included:")
            print("  1. Time-Space Context")
            print("  2. Character History")
            print("  3. Emotional Expression")
            print("  4. Inner Thought Analysis")
            print("  5. Conversational Dynamics")
            print("  6. Motivational Insights")
        
        except Exception as e:
            print(f"Error saving processed data: {e}")
    
    @classmethod
    def create_preprocessing_pipeline(
        cls,
        novel_path: str,
        anthropic_api_key: Optional[str] = None
    ) -> 'CharacterBehaviorDataPreprocessor':
        """
        Factory method to create a comprehensive preprocessing pipeline
        
        Args:
            novel_path (str): Path to the novel text file
            anthropic_api_key (Optional[str]): API key for enhanced feature generation
        
        Returns:
            Configured CharacterBehaviorDataPreprocessor instance
        """
        # Read novel text
        with open(novel_path, 'r', encoding='utf-8') as f:
            novel_text = f.read()
        
        # Initialize Anthropic client if API key provided
        anthropic_client = None
        if anthropic_api_key:
            anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # Create preprocessor instance
        preprocessor = cls(
            novel_text,
            anthropic_client=anthropic_client
        )
        
        return preprocessor

# Comprehensive Preprocessing Workflow Example
def main():
    # Configuration
    NOVEL_PATH = "anne_of_green_gables.txt"
    OUTPUT_PATH = "processed_anne_of_green_gables.json"
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Optional
    
    try:
        # Create preprocessing pipeline
        preprocessor = CharacterBehaviorDataPreprocessor.create_preprocessing_pipeline(
            novel_path=NOVEL_PATH,
            anthropic_api_key=ANTHROPIC_API_KEY
        )
        
        # Execute full preprocessing
        preprocessor.save_processed_data(OUTPUT_PATH)
    
    except Exception as e:
        print(f"Preprocessing failed: {e}")

if __name__ == "__main__":
    main()

"""
Preprocessing Pipeline Design Principles:

1. Modular Architecture
   - Separates concerns across different preprocessing stages
   - Allows easy extension and modification

2. Robust Error Handling
   - Graceful handling of processing failures
   - Provides comprehensive logging
   - Ensures pipeline continues even if individual scenes fail

3. Metadata-Enhanced Feature Extraction
   - Goes beyond simple text processing
   - Integrates computational linguistics with scholarly insights
   - Provides multi-dimensional narrative understanding

4. Flexible Configuration
   - Supports optional AI-assisted feature generation
   - Adaptable to different novels and preprocessing requirements

Future Enhancement Directions:
- More sophisticated scene boundary detection
- Advanced metadata integration techniques
- Machine learning-based feature validation
- Support for multiple languages and narrative styles
"""# Character Behavior Modeling: Data Preprocessing Pipeline

import json
import re
import torch
import transformers
import anthropic
import spacy
import nltk
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Download necessary NLP resources
nltk.download('punkt')
spacy.cli.download("en_core_web_sm")

class CharacterBehaviorDataPreprocessor:
    """
    Comprehensive data preprocessing system for transforming 
    literary narrative into expert-ready training tokens
    """
    
    def __init__(
        self, 
        novel_text: str,
        anthropic_api_key: str = None,
        tokenizer: transformers.PreTrainedTokenizer = None
    ):
        # Core text processing components
        self.novel_text = novel_text
        self.nlp = spacy.load("en_core_web_sm")
        
        # Tokenization setup
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained("facebook/opt-350m")
        
        # Optional AI-assisted data generation
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
    
    def split_into_scenes(
        self, 
        max_scene_tokens: int = 1024, 
        min_scene_tokens: int = 128
    ) -> List[str]:
        """
        Hybrid scene segmentation using NLTK and SpaCy
        
        Segmentation Strategies:
        1. Use TextTiling for initial discourse boundaries
        2. Refine with SpaCy linguistic analysis
        3. Maintain contextual integrity
        4. Respect token length constraints
        """
        import nltk
        from nltk.tokenize import TextTilingTokenizer
        
        # Ensure NLTK resources are downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Initial TextTiling segmentation
        text_tiler = TextTilingTokenizer(
            w=20,    # Paragraph width
            k=10     # Number of sentences for block comparison
        )
        
        # Perform initial segmentation
        initial_segments = text_tiler.tokenize(self.novel_text)
        
        # Refine segments using SpaCy
        scenes = []
        current_scene = []
        current_scene_tokens = 0
        
        for segment in initial_segments:
            # Process segment with SpaCy for additional linguistic analysis
            doc = self.nlp(segment)
            
            # Tokenize the segment
            segment_tokens = self.tokenizer.encode(segment)
            segment_token_count = len(segment_tokens)
            
            # Scene boundary management
            if current_scene_tokens + segment_token_count > max_scene_tokens:
                # Commit current scene if it meets minimum token threshold
                if current_scene_tokens >= min_scene_tokens:
                    scenes.append(' '.join(current_scene))
                
                # Reset for new scene
                current_scene = [segment]
                current_scene_tokens = segment_token_count
            else:
                current_scene.append(segment)
                current_scene_tokens += segment_token_count
        
        # Add final scene if it meets minimum token threshold
        if current_scene and current_scene_tokens >= min_scene_tokens:
            scenes.append(' '.join(current_scene))
        
        return scenes
    
    @dataclass
    class ExpertFeatures:
        """
        Structured representation of scene features for each expert
        """
        time_space: Dict[str, Any]
        character_history: Dict[str, Any]
        expression: Dict[str, Any]
        inner_thought: Dict[str, Any]
        conversation: Dict[str, Any]
        motivation: Dict[str, Any]
    
    def extract_scene_features(self, scene: str) -> ExpertFeatures:
        """
        Advanced multi-dimensional scene feature extraction
        
        Leverages state-of-the-art NLP and machine learning libraries
        to comprehensively analyze narrative scenes
        
        Args:
            scene (str): Narrative scene text to process
        
        Returns:
            ExpertFeatures: Structured features for each expert domain
        """
        # SpaCy linguistic processing
        doc = self.nlp(scene)
        
        # Time-Space Expert Feature Extraction
        def extract_time_space_features(text):
            """
            Comprehensive spatial and temporal feature extraction
            
            Combines multiple libraries for robust coordinate identification
            """
            import spacy
            from geopy.geocoders import Nominatim
            
            # SpaCy entity extraction
            entities = {
                'locations': [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC']],
                'temporal_markers': [token.text for token in doc if token.pos_ == 'TIME']
            }
            
            # Geocoding for location coordinates
            geolocator = Nominatim(user_agent="character_behavior_model")
            coordinates = {}
            for location in entities['locations']:
                try:
                    geo_location = geolocator.geocode(location)
                    if geo_location:
                        coordinates[location] = {
                            'latitude': geo_location.latitude,
                            'longitude': geo_location.longitude
                        }
                except Exception:
                    # Graceful handling of geocoding failures
                    coordinates[location] = None
            
            return {
                'named_entities': entities,
                'geographical_coordinates': coordinates
            }
        
        # Character History Feature Extraction
        def extract_character_history_features(text):
            """
            Advanced character relationship and reference tracking
            
            Uses network analysis and coreference resolution
            """
            import networkx as nx
            
            # Extract character mentions
            characters = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
            
            # Build character interaction graph
            interaction_graph = nx.Graph()
            for char in characters:
                interaction_graph.add_node(char)
            
            # Simple co-occurrence based relationship tracking
            for i in range(len(characters)):
                for j in range(i+1, len(characters)):
                    interaction_graph.add_edge(characters[i], characters[j])
            
            return {
                'characters_mentioned': characters,
                'interaction_graph': nx.node_link_data(interaction_graph)
            }
        
        # Expression Feature Extraction
        def extract_expression_features(text):
            """
            Multi-dimensional emotional and expression analysis
            
            Combines NLTK and VADER for comprehensive sentiment detection
            """
            from nltk.sentiment import SentimentIntensityAnalyzer
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderSentiment
            
            # NLTK Sentiment Analysis
            nltk_analyzer = SentimentIntensityAnalyzer()
            nltk_sentiment = nltk_analyzer.polarity_scores(text)
            
            # VADER Sentiment Analysis
            vader_analyzer = VaderSentiment()
            vader_sentiment = vader_analyzer.polarity_scores(text)
            
            # Emotional expression markers
            emotion_keywords = [
                token.text for token in doc 
                if token.text.lower() in [
                    'happy', 'sad', 'angry', 'excited', 
                    'frustrated', 'delighted', 'worried'
                ]
            ]
            
            return {
                'nltk_sentiment': nltk_sentiment,
                'vader_sentiment': vader_sentiment,
                'emotion_keywords': emotion_keywords
            }
        
        # Inner Thought Feature Extraction
        def extract_inner_thought_features(text):
            """
            Introspective and psychological state analysis
            
            Identifies internal reflection structures and psychological markers
            """
            # Introspective linguistic markers
            introspective_markers = [
                sent.text for sent in doc.sents 
                if any(token.text.lower() in ['thought', 'wondered', 'imagined', 'felt'] 
                       for token in sent)
            ]
            
            # Dependency structure analysis for psychological insight
            psychological_structures = [
                {
                    'root': token.head.text,
                    'dependency': token.dep_,
                    'text': token.text
                } 
                for token in doc 
                if token.dep_ in ['xcomp', 'ccomp', 'advcl']  # Clauses with potential psychological depth
            ]
            
            return {
                'introspective_markers': introspective_markers,
                'psychological_structures': psychological_structures
            }
        
        # Conversation Feature Extraction
        def extract_conversation_features(text):
            """
            Dialogue structure and conversational dynamics analysis
            
            Identifies dialogue segments and interaction patterns
            """
            # Dialogue segment identification
            dialogue_segments = [
                sent.text for sent in doc.sents 
                if '"' in sent.text  # Simple dialogue marker
            ]
            
            # Speaker identification
            speakers = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
            
            return {
                'dialogue_segments': dialogue_segments,
                'potential_speakers': speakers
            }
        
        # Motivation Feature Extraction
        def extract_motivation_features(text):
            """
            Psychological motivation and goal identification
            
            Analyzes linguistic markers of underlying motivations
            """
            # Motivation-related keyword detection
            motivation_keywords = [
                token.text for token in doc 
                if token.text.lower() in [
                    'want', 'need', 'desire', 'hope', 'dream', 
                    'wish', 'goal', 'aspiration'
                ]
            ]
            
            # Contextual motivation analysis
            motivation_context = [
                sent.text for sent in doc.sents
                if any(keyword in sent.text.lower() for keyword in [
                    'want', 'need', 'desire', 'hope', 'dream'
                ])
            ]
            
            return {
                'motivation_keywords': motivation_keywords,
                'motivation_context': motivation_context
            }
        
        # Comprehensive feature extraction
        return self.ExpertFeatures(
            time_space=extract_time_space_features(scene),
            character_history=extract_character_history_features(scene),
            expression=extract_expression_features(scene),
            inner_thought=extract_inner_thought_features(scene),
            conversation=extract_conversation_features(scene),
            motivation=extract_motivation_features(scene)
        )
    
    def augment_scene_features(
        self, 
        scene_features: Dict[str, Any], 
        scene_context: str
    ) -> Dict[str, Any]:
        """
        Advanced AI-assisted feature augmentation and completion
        
        Comprehensive approach to filling gaps in scene feature extraction
        
        Args:
            scene_features (Dict): Existing scene features from extraction
            scene_context (str): Full text context of the scene
        
        Returns:
            Dict: Augmented and completed scene features
        """
        # Validate input features
        if not scene_features or not scene_context:
            return scene_features
        
        # Prepare augmentation prompt
        augmentation_prompt = self._construct_augmentation_prompt(
            scene_features, 
            scene_context
        )
        
        # Attempt AI-assisted feature augmentation
        try:
            augmented_features = self._generate_ai_augmentations(
                augmentation_prompt, 
                scene_features
            )
            
            # Merge and validate augmented features
            final_features = self._merge_and_validate_features(
                scene_features, 
                augmented_features
            )
            
            return final_features
        
        except Exception as e:
            print(f"Feature augmentation error: {e}")
            return scene_features
    
    def _construct_augmentation_prompt(
        self, 
        scene_features: Dict[str, Any], 
        scene_context: str
    ) -> str:
        """
        Create a sophisticated prompt for feature augmentation
        
        Provides context and highlights missing or incomplete dimensions
        
        Args:
            scene_features (Dict): Current scene features
            scene_context (str): Full scene text
        
        Returns:
            str: Comprehensive augmentation prompt
        """
        # Identify missing or potentially incomplete feature dimensions
        missing_dimensions = self._identify_missing_dimensions(scene_features)
        
        # Construct detailed augmentation prompt
        prompt = f"""
        Analyze the following narrative scene and provide comprehensive insights 
        to complete missing or implicit feature dimensions.

        Scene Context:
        ```
        {scene_context}
        ```

        Current Feature Dimensions:
        {json.dumps(scene_features, indent=2)}

        Missing or Incomplete Dimensions to Explore:
        {', '.join(missing_dimensions)}

        For each missing dimension, provide:
        1. Detailed contextual analysis
        2. Inferred feature values
        3. Confidence level of inference
        4. Reasoning behind the inference

        Respond in a structured JSON format that can be directly merged with existing features.
        Focus on capturing nuanced, character-specific insights.
        """
        
        return prompt
    
    def _identify_missing_dimensions(
        self, 
        scene_features: Dict[str, Any]
    ) -> List[str]:
        """
        Identify potentially incomplete or missing feature dimensions
        
        Args:
            scene_features (Dict): Current scene features
        
        Returns:
            List[str]: Dimensions requiring augmentation
        """
        missing_dimensions = []
        
        # Time-Space Expert Completeness Check
        if not scene_features.get('time_space', {}).get('geographical_coordinates'):
            missing_dimensions.append('Geographical Context')
        
        # Character History Completeness
        if not scene_features.get('character_history', {}).get('character_interactions'):
            missing_dimensions.append('Character Relationship Dynamics')
        
        # Expression Depth Check
        if not scene_features.get('expression', {}).get('emotional_intensity'):
            missing_dimensions.append('Emotional Nuance')
        
        # Inner Thought Complexity
        if not scene_features.get('inner_thought', {}).get('psychological_structures'):
            missing_dimensions.append('Psychological Depth')
        
        # Conversation Richness
        if not scene_features.get('conversation', {}).get('dialogue_context'):
            missing_dimensions.append('Conversational Subtext')
        
        # Motivation Comprehensiveness
        if not scene_features.get('motivation', {}).get('underlying_drives'):
            missing_dimensions.append('Motivational Drivers')
        
        return missing_dimensions
    
    def _generate_ai_augmentations(
        self, 
        prompt: str, 
        existing_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate AI-assisted feature augmentations
        
        Uses Anthropic API to intelligently complete feature dimensions
        
        Args:
            prompt (str): Comprehensive augmentation prompt
            existing_features (Dict): Current scene features
        
        Returns:
            Dict: AI-generated feature augmentations
        """
        # Validate Anthropic client availability
        if not self.anthropic_client:
            raise ValueError("Anthropic API client not configured")
        
        # Generate AI augmentation
        try:
            completion = self.anthropic_client.completions.create(
                model="claude-2",
                prompt=prompt,
                max_tokens_to_sample=1000,
                stop_sequences=["```"]
            )
            
            # Parse AI-generated features
            augmented_features = json.loads(completion.completion)
            
            return augmented_features
        
        except Exception as e:
            print(f"AI augmentation generation failed: {e}")
            raise
    
    def _merge_and_validate_features(
        self, 
        original_features: Dict[str, Any], 
        augmented_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge and validate AI-augmented features
        
        Ensures feature integrity and prevents over-augmentation
        
        Args:
            original_features (Dict): Original scene features
            augmented_features (Dict): AI-generated feature augmentations
        
        Returns:
            Dict: Merged and validated feature set
        """
        # Deep copy to prevent direct modification
        merged_features = copy.deepcopy(original_features)
        
        # Merge augmented features with conservative approach
        for expert_domain, domain_features in augmented_features.items():
            if expert_domain in merged_features:
                # Merge only if augmented features provide additional insights
                for key, value in domain_features.items():
                    if not merged_features[expert_domain].get(key):
                        merged_features[expert_domain][key] = value
        
        return merged_features
    
    def process_novel(self) -> List[Dict]:
        """
        Full novel preprocessing pipeline with AI-assisted augmentation
        
        Returns:
        - List of scene tokens, each with comprehensive, augmented features
        """
        # Scene segmentation
        scenes = self.split_into_scenes()
        
        # Processed scenes with feature extraction and augmentation
        processed_scenes = []
        
        for scene in scenes:
            # Extract initial features
            initial_features = asdict(self.extract_scene_features(scene))
            
            # AI-assisted feature augmentation
            augmented_features = self.augment_scene_features(
                initial_features, 
                scene
            )
            
            processed_scenes.append(augmented_features)
        
        return processed_scenes
    
    def save_processed_data(
        self, 
        output_path: str = "anne_of_green_gables_processed.json"
    ):
        """
        Save processed novel data to JSON
        """
        processed_data = self.process_novel()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"Processed data saved to {output_path}")

# Usage Example
def main():
    # Load novel text (placeholder - replace with actual file reading)
    with open('anne_of_green_gables.txt', 'r', encoding='utf-8') as f:
        novel_text = f.read()
    
    preprocessor = CharacterBehaviorDataPreprocessor(
        novel_text,
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')  # Optional
    )
    
    preprocessor.save_processed_data()

if __name__ == "__main__":
    main()

"""
Preprocessing Strategy Highlights:

1. Scene Segmentation
   - Intelligent scene boundary detection
   - Respects contextual and token-length constraints
   - Maintains narrative coherence

2. Multi-Expert Feature Extraction
   - Specialized extraction for each expert
   - NLP-powered feature identification
   - Handles multiple feature dimensions

3. AI-Assisted Feature Augmentation
   - Fallback mechanism for missing or implicit features
   - Uses Anthropic API for intelligent feature generation
   - Ensures comprehensive feature coverage

4. Flexible and Extensible Design
   - Easy to add new feature extraction methods
   - Supports different tokenization strategies
   - Adaptable to various narrative structures

Potential Future Enhancements:
- Machine learning-based scene boundary detection
- More sophisticated NLP feature extraction
- Advanced AI feature augmentation
"""
