"""
Language detection for text content.

This module provides language detection capabilities to help select
appropriate chunking strategies based on the language of the text content.
"""

import logging
import re
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Detects the language of text content.

    Uses simple heuristics and pattern matching to detect common languages.
    Can be extended with more sophisticated language detection libraries.
    """

    def __init__(self):
        """Initialize language detector."""
        self.logger = logging.getLogger(f"{__name__}.LanguageDetector")

        # Simple language patterns for common languages
        self.language_patterns = {
            'en': {
                'common_words': ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'],
                'patterns': [r'\bthe\b', r'\band\b', r'\bof\b', r'\bto\b', r'\ba\b'],
                'name': 'English'
            },
            'es': {
                'common_words': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo'],
                'patterns': [r'\bel\b', r'\bla\b', r'\bde\b', r'\bque\b', r'\by\b'],
                'name': 'Spanish'
            },
            'fr': {
                'common_words': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour'],
                'patterns': [r'\ble\b', r'\bde\b', r'\bet\b', r'\bà\b', r'\bun\b'],
                'name': 'French'
            },
            'de': {
                'common_words': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf'],
                'patterns': [r'\bder\b', r'\bdie\b', r'\bund\b', r'\bin\b', r'\bden\b'],
                'name': 'German'
            },
            'it': {
                'common_words': ['il', 'di', 'che', 'e', 'la', 'per', 'un', 'in', 'con', 'da', 'su', 'come'],
                'patterns': [r'\bil\b', r'\bdi\b', r'\bche\b', r'\be\b', r'\bla\b'],
                'name': 'Italian'
            },
            'pt': {
                'common_words': ['o', 'de', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma'],
                'patterns': [r'\bo\b', r'\bde\b', r'\be\b', r'\bdo\b', r'\bda\b'],
                'name': 'Portuguese'
            },
            'ru': {
                'common_words': ['в', 'и', 'не', 'на', 'я', 'быть', 'он', 'с', 'а', 'как', 'это', 'вы'],
                'patterns': [r'\bв\b', r'\bи\b', r'\bне\b', r'\bна\b', r'\bя\b'],
                'name': 'Russian'
            },
            'zh': {
                'common_words': ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一'],
                'patterns': [r'的', r'了', r'在', r'是', r'我'],
                'name': 'Chinese'
            },
            'ja': {
                'common_words': ['の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ'],
                'patterns': [r'の', r'に', r'は', r'を', r'た'],
                'name': 'Japanese'
            },
            'ar': {
                'common_words': ['في', 'من', 'على', 'إلى', 'عن', 'مع', 'هذا', 'هذه', 'التي', 'الذي', 'كان', 'لم'],
                'patterns': [r'في', r'من', r'على', r'إلى', r'عن'],
                'name': 'Arabic'
            }
        }

        # Character ranges for different scripts
        self.script_ranges = {
            'latin': (0x0020, 0x024F),
            'cyrillic': (0x0400, 0x04FF),
            'arabic': (0x0600, 0x06FF),
            'cjk': (0x4E00, 0x9FFF),  # Chinese, Japanese, Korean
            'hiragana': (0x3040, 0x309F),
            'katakana': (0x30A0, 0x30FF),
            'hangul': (0xAC00, 0xD7AF),
        }

    def detect(self, text: str, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Detect language of text content.

        Args:
            text: Text to analyze
            sample_size: Number of characters to sample for analysis

        Returns:
            Dictionary with language detection results
        """
        if not text or not text.strip():
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'detected_scripts': [],
                'method': 'none'
            }

        # Sample text for analysis
        sample_text = text[:sample_size].lower()

        # Detect scripts first
        detected_scripts = self._detect_scripts(text)

        # Try pattern-based detection
        language_scores = {}
        for lang_code, lang_info in self.language_patterns.items():
            score = self._calculate_language_score(sample_text, lang_info)
            if score > 0:
                language_scores[lang_code] = score

        # Determine best match
        if language_scores:
            best_language = max(language_scores, key=language_scores.get)
            confidence = language_scores[best_language]
            method = 'pattern_matching'
        else:
            # Fallback to script-based detection
            best_language = self._script_to_language(detected_scripts)
            confidence = 0.3 if best_language != 'unknown' else 0.0
            method = 'script_detection'

        # Try external library if available
        external_result = self._try_external_detection(text)
        if external_result and external_result['confidence'] > confidence:
            best_language = external_result['language']
            confidence = external_result['confidence']
            method = 'external_library'

        return {
            'language': best_language,
            'language_name': self.language_patterns.get(best_language, {}).get('name', best_language),
            'confidence': confidence,
            'detected_scripts': detected_scripts,
            'method': method,
            'alternative_languages': sorted(language_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        }

    def detect_file(self, file_path: Union[str, Path], encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        Detect language of text file.

        Args:
            file_path: Path to text file
            encoding: Text encoding to use

        Returns:
            Dictionary with language detection results
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                # Read first few KB for detection
                text = f.read(8192)

            result = self.detect(text)
            result['file_path'] = str(file_path)
            result['encoding_used'] = encoding

            return result

        except Exception as e:
            self.logger.error(f"Error detecting language in file {file_path}: {e}")
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'error': str(e),
                'file_path': str(file_path)
            }

    def is_supported_language(self, language_code: str) -> bool:
        """Check if language is supported by the detector."""
        return language_code in self.language_patterns

    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages."""
        return {code: info['name'] for code, info in self.language_patterns.items()}

    def get_chunking_suggestions(self, language: str) -> Dict[str, Any]:
        """
        Get chunking strategy suggestions for a specific language.

        Args:
            language: Detected language code

        Returns:
            Dictionary with strategy suggestions
        """
        suggestions = {
            'sentence_strategies': [],
            'word_strategies': [],
            'character_strategies': [],
            'special_considerations': []
        }

        # Language-specific suggestions
        if language in ['zh', 'ja']:  # Chinese, Japanese
            suggestions.update({
                'sentence_strategies': ['character_based', 'semantic_basic'],
                'word_strategies': ['character_based'],  # No clear word boundaries
                'character_strategies': ['character_based', 'fixed_size'],
                'special_considerations': [
                    'No clear word boundaries',
                    'Consider character-based chunking',
                    'Semantic chunking may work better'
                ]
            })

        elif language == 'ar':  # Arabic
            suggestions.update({
                'sentence_strategies': ['sentence_based', 'boundary_aware'],
                'word_strategies': ['word_fixed_length', 'boundary_aware'],
                'character_strategies': ['character_based', 'fixed_size'],
                'special_considerations': [
                    'Right-to-left text direction',
                    'Connected script considerations',
                    'Diacritics handling'
                ]
            })

        elif language == 'de':  # German
            suggestions.update({
                'sentence_strategies': ['sentence_based', 'compound_aware'],
                'word_strategies': ['word_fixed_length', 'compound_aware'],
                'character_strategies': ['character_based'],
                'special_considerations': [
                    'Long compound words',
                    'Consider splitting compounds',
                    'Capitalization rules'
                ]
            })

        elif language in ['en', 'fr', 'es', 'it', 'pt']:  # Romance/Germanic languages
            suggestions.update({
                'sentence_strategies': ['sentence_based', 'paragraph_based'],
                'word_strategies': ['word_fixed_length', 'token_based'],
                'character_strategies': ['character_based', 'fixed_size'],
                'special_considerations': [
                    'Standard sentence boundaries',
                    'Clear word boundaries',
                    'Punctuation-based chunking works well'
                ]
            })

        elif language == 'ru':  # Russian
            suggestions.update({
                'sentence_strategies': ['sentence_based', 'boundary_aware'],
                'word_strategies': ['word_fixed_length', 'morphological_aware'],
                'character_strategies': ['character_based'],
                'special_considerations': [
                    'Cyrillic script',
                    'Complex morphology',
                    'Case system considerations'
                ]
            })

        else:  # Unknown or unsupported language
            suggestions.update({
                'sentence_strategies': ['sentence_based', 'boundary_aware'],
                'word_strategies': ['word_fixed_length', 'character_based'],
                'character_strategies': ['character_based', 'fixed_size'],
                'special_considerations': [
                    'Language-specific features unknown',
                    'Use generic approaches',
                    'Monitor chunking quality'
                ]
            })

        return suggestions

    def _detect_scripts(self, text: str) -> List[str]:
        """Detect writing scripts in text."""
        scripts = []
        char_counts = {script: 0 for script in self.script_ranges}

        for char in text:
            char_code = ord(char)
            for script, (start, end) in self.script_ranges.items():
                if start <= char_code <= end:
                    char_counts[script] += 1
                    break

        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return scripts

        # Include scripts that make up at least 5% of characters
        for script, count in char_counts.items():
            if count / total_chars >= 0.05:
                scripts.append(script)

        return scripts

    def _calculate_language_score(self, text: str, lang_info: Dict[str, Any]) -> float:
        """Calculate language score based on patterns and common words."""
        score = 0.0

        # Count common words
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0

        common_word_count = sum(1 for word in words if word in lang_info['common_words'])
        word_score = common_word_count / len(words)

        # Count pattern matches
        pattern_matches = 0
        for pattern in lang_info['patterns']:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            pattern_matches += matches

        pattern_score = min(pattern_matches / len(text) * 1000, 1.0)  # Normalize

        # Combine scores
        score = (word_score * 0.7) + (pattern_score * 0.3)

        return score

    def _script_to_language(self, scripts: List[str]) -> str:
        """Map detected scripts to likely languages."""
        if not scripts:
            return 'unknown'

        # Simple mapping of scripts to languages
        script_language_map = {
            'latin': 'en',  # Default to English for Latin script
            'cyrillic': 'ru',
            'arabic': 'ar',
            'cjk': 'zh',
            'hiragana': 'ja',
            'katakana': 'ja',
            'hangul': 'ko'
        }

        # Return the first mapped language
        for script in scripts:
            if script in script_language_map:
                return script_language_map[script]

        return 'unknown'

    def _try_external_detection(self, text: str) -> Optional[Dict[str, Any]]:
        """Try external language detection library if available."""
        try:
            # Try langdetect library
            import langdetect
            detected = langdetect.detect_langs(text)
            if detected:
                best = detected[0]
                return {
                    'language': best.lang,
                    'confidence': best.prob
                }
        except ImportError:
            pass
        except Exception as e:
            self.logger.debug(f"External language detection failed: {e}")

        try:
            # Try polyglot library
            from polyglot.detect import Detector
            detector = Detector(text)
            return {
                'language': detector.language.code,
                'confidence': detector.language.confidence / 100.0
            }
        except ImportError:
            pass
        except Exception as e:
            self.logger.debug(f"Polyglot language detection failed: {e}")

        return None
