"""
Content analysis for informed chunking strategy selection.

This module analyzes content characteristics to help select the most
appropriate chunking strategies and parameters.
"""

import logging
import math
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import statistics

from chunking_strategy.core.base import ModalityType
from chunking_strategy.detectors.language_detector import LanguageDetector
from chunking_strategy.detectors.encoding_detector import EncodingDetector

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """
    Analyzes content characteristics for chunking optimization.

    Provides insights about content structure, complexity, and characteristics
    that can inform chunking strategy selection and parameter tuning.
    """

    def __init__(self):
        """Initialize content analyzer."""
        self.logger = logging.getLogger(f"{__name__}.ContentAnalyzer")
        self.language_detector = LanguageDetector()
        self.encoding_detector = EncodingDetector()

    def analyze_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze a file's content characteristics.

        Args:
            file_path: Path to the file to analyze

        Returns:
            Dictionary with content analysis results
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check if file extension indicates binary format
        extension = file_path.suffix.lower()
        binary_extensions = {
            # Audio
            '.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma', '.opus',
            # Video
            '.mp4', '.avi', '.mkv', '.webm', '.flv', '.mov',
            # Images
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff',
            # Archives & binary
            '.zip', '.tar', '.gz', '.7z', '.rar', '.exe', '.bin'
        }

        # For large files, only read a sample for analysis (max 1MB)
        MAX_ANALYSIS_SIZE = 1024 * 1024  # 1MB
        file_size = file_path.stat().st_size

        if extension in binary_extensions:
            # Direct binary analysis for known binary formats
            if file_size > MAX_ANALYSIS_SIZE:
                # Read only first 1MB for analysis
                with open(file_path, 'rb') as f:
                    content = f.read(MAX_ANALYSIS_SIZE)
            else:
                content = file_path.read_bytes()
            modality = self._guess_modality_from_bytes(content)
            encoding_info = {'encoding': 'binary'}
        else:
            # Try text analysis first for other files
            encoding_info = self.encoding_detector.detect(file_path)
            encoding = encoding_info['encoding']

            try:
                if file_size > MAX_ANALYSIS_SIZE:
                    # Read only first 1MB for analysis
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read(MAX_ANALYSIS_SIZE)
                else:
                    content = file_path.read_text(encoding=encoding)
                modality = ModalityType.TEXT
            except (UnicodeDecodeError, UnicodeError):
                # Fallback to binary analysis
                if file_size > MAX_ANALYSIS_SIZE:
                    # Read only first 1MB for analysis
                    with open(file_path, 'rb') as f:
                        content = f.read(MAX_ANALYSIS_SIZE)
                else:
                    content = file_path.read_bytes()
                modality = self._guess_modality_from_bytes(content)

        result = self.analyze_content(content)
        result.update({
            'file_path': str(file_path),
            'encoding': encoding_info['encoding'],
            'modality': modality
        })

        return result

    def analyze_content(
        self,
        content: Union[str, bytes],
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze content characteristics.

        Args:
            content: Content to analyze
            content_type: MIME type hint

        Returns:
            Dictionary with analysis results
        """
        if isinstance(content, bytes):
            return self._analyze_binary_content(content, content_type)
        else:
            return self._analyze_text_content(content, content_type)

    def get_chunking_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get chunking recommendations based on content analysis.

        Args:
            analysis: Content analysis results

        Returns:
            Dictionary with chunking recommendations
        """
        recommendations = {
            'primary_strategies': [],
            'secondary_strategies': [],
            'parameters': {},
            'considerations': []
        }

        modality = analysis.get('modality', ModalityType.TEXT)

        if modality == ModalityType.TEXT:
            recommendations.update(self._get_text_recommendations(analysis))
        elif modality == ModalityType.IMAGE:
            recommendations.update(self._get_image_recommendations(analysis))
        elif modality == ModalityType.AUDIO:
            recommendations.update(self._get_audio_recommendations(analysis))
        elif modality == ModalityType.VIDEO:
            recommendations.update(self._get_video_recommendations(analysis))
        else:
            recommendations.update(self._get_generic_recommendations(analysis))

        return recommendations

    def _analyze_text_content(self, content: str, content_type: Optional[str] = None) -> Dict[str, Any]:
        """Analyze text content characteristics."""
        if not content.strip():
            return {
                'modality': ModalityType.TEXT,
                'content_length': 0,
                'analysis_method': 'empty_content'
            }

        # Basic statistics
        char_count = len(content)
        line_count = content.count('\n') + 1
        word_count = len(re.findall(r'\b\w+\b', content))

        # Language detection
        language_info = self.language_detector.detect(content)

        # Structure analysis
        structure = self._analyze_text_structure(content)

        # Complexity analysis
        complexity = self._analyze_text_complexity(content)

        return {
            'modality': ModalityType.TEXT,
            'content_length': char_count,
            'line_count': line_count,
            'word_count': word_count,
            'avg_line_length': char_count / line_count if line_count > 0 else 0,
            'avg_word_length': sum(len(word) for word in re.findall(r'\b\w+\b', content)) / word_count if word_count > 0 else 0,
            'language': language_info['language'],
            'language_confidence': language_info['confidence'],
            'structure': structure,
            'complexity': complexity,
            'analysis_method': 'text_analysis'
        }

    def _analyze_binary_content(self, content: bytes, content_type: Optional[str] = None) -> Dict[str, Any]:
        """Analyze binary content characteristics."""
        size = len(content)

        # Basic entropy calculation
        entropy = self._calculate_entropy(content)

        # Guess modality
        modality = self._guess_modality_from_bytes(content)

        return {
            'modality': modality,
            'content_size': size,
            'entropy': entropy,
            'compression_potential': self._estimate_compression_potential(content),
            'analysis_method': 'binary_analysis'
        }

    def _analyze_text_structure(self, content: str) -> Dict[str, Any]:
        """Analyze text structure characteristics."""
        # Paragraph detection
        paragraphs = re.split(r'\n\s*\n', content)
        paragraph_count = len([p for p in paragraphs if p.strip()])

        # Sentence detection
        sentences = re.split(r'[.!?]+', content)
        sentence_count = len([s for s in sentences if s.strip()])

        # Heading detection
        heading_patterns = [
            r'^#+\s+',  # Markdown headings
            r'^[A-Z][A-Z\s]+$',  # All caps lines
            r'^\d+\.\s+',  # Numbered sections
        ]

        heading_count = 0
        for pattern in heading_patterns:
            heading_count += len(re.findall(pattern, content, re.MULTILINE))

        # List detection
        list_items = len(re.findall(r'^\s*[-*+]\s+', content, re.MULTILINE))
        list_items += len(re.findall(r'^\s*\d+\.\s+', content, re.MULTILINE))

        # Code detection
        code_blocks = len(re.findall(r'```[\s\S]*?```', content))
        inline_code = len(re.findall(r'`[^`]+`', content))

        return {
            'paragraph_count': paragraph_count,
            'sentence_count': sentence_count,
            'heading_count': heading_count,
            'list_item_count': list_items,
            'code_block_count': code_blocks,
            'inline_code_count': inline_code,
            'avg_paragraph_length': len(content) / paragraph_count if paragraph_count > 0 else 0,
            'avg_sentence_length': len(content) / sentence_count if sentence_count > 0 else 0,
            'structure_type': self._classify_structure_type(paragraph_count, heading_count, list_items, code_blocks)
        }

    def _analyze_text_complexity(self, content: str) -> Dict[str, Any]:
        """Analyze text complexity metrics."""
        words = re.findall(r'\b\w+\b', content.lower())

        if not words:
            return {'complexity_score': 0.0, 'readability': 'unknown'}

        # Vocabulary diversity
        unique_words = len(set(words))
        total_words = len(words)
        vocabulary_diversity = unique_words / total_words if total_words > 0 else 0

        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Sentence complexity
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

        # Punctuation density
        punctuation_count = len(re.findall(r'[.,;:!?]', content))
        punctuation_density = punctuation_count / len(content) if len(content) > 0 else 0

        # Simple complexity score
        complexity_score = (
            min(avg_word_length / 10, 1.0) * 0.3 +
            min(avg_sentence_length / 30, 1.0) * 0.3 +
            vocabulary_diversity * 0.2 +
            min(punctuation_density * 100, 1.0) * 0.2
        )

        return {
            'complexity_score': complexity_score,
            'vocabulary_diversity': vocabulary_diversity,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'punctuation_density': punctuation_density,
            'readability': self._classify_readability(complexity_score)
        }

    def _classify_structure_type(self, paragraphs: int, headings: int, lists: int, code: int) -> str:
        """Classify the type of text structure."""
        if code > 0:
            return 'technical_document'
        elif headings > paragraphs * 0.1:
            return 'structured_document'
        elif lists > paragraphs * 0.2:
            return 'list_heavy'
        elif paragraphs > 10 and headings < 3:
            return 'narrative'
        else:
            return 'mixed'

    def _classify_readability(self, complexity_score: float) -> str:
        """Classify readability level."""
        if complexity_score < 0.3:
            return 'simple'
        elif complexity_score < 0.6:
            return 'moderate'
        else:
            return 'complex'

    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate entropy of binary data."""
        if not data:
            return 0.0

        # Count byte frequencies
        frequencies = [0] * 256
        for byte in data:
            frequencies[byte] += 1

        # Calculate entropy
        entropy = 0.0
        data_len = len(data)

        for freq in frequencies:
            if freq > 0:
                probability = freq / data_len
                entropy -= probability * math.log2(probability)

        return entropy

    def _estimate_compression_potential(self, data: bytes) -> float:
        """Estimate compression potential of binary data."""
        if not data:
            return 0.0

        # Simple heuristic based on byte diversity
        unique_bytes = len(set(data))
        max_unique = min(256, len(data))

        diversity = unique_bytes / max_unique

        # Higher diversity suggests better compression potential
        return 1.0 - diversity

    def _guess_modality_from_bytes(self, content: bytes) -> ModalityType:
        """Guess modality from byte content."""
        if not content:
            return ModalityType.TEXT

        # Check for common file signatures

        # Image formats
        if content.startswith(b'\x89PNG'):
            return ModalityType.IMAGE
        elif content.startswith(b'\xFF\xD8\xFF'):  # JPEG
            return ModalityType.IMAGE
        elif content.startswith(b'GIF87a') or content.startswith(b'GIF89a'):
            return ModalityType.IMAGE
        elif content.startswith(b'BM'):  # BMP
            return ModalityType.IMAGE
        elif content.startswith(b'RIFF') and b'WEBP' in content[:12]:
            return ModalityType.IMAGE

        # Audio formats
        elif content.startswith(b'RIFF') and b'WAVE' in content[:12]:  # WAV
            return ModalityType.AUDIO
        elif content.startswith(b'ID3') or content.startswith(b'\xFF\xFB') or content.startswith(b'\xFF\xF3'):  # MP3
            return ModalityType.AUDIO
        elif content.startswith(b'OggS'):  # OGG/Vorbis/Opus
            return ModalityType.AUDIO
        elif content.startswith(b'fLaC'):  # FLAC
            return ModalityType.AUDIO
        elif content.startswith(b'\x00\x00\x00\x20ftypM4A') or content.startswith(b'\x00\x00\x00\x18ftypmp42'):  # M4A/AAC
            return ModalityType.AUDIO

        # Video formats
        elif content.startswith(b'\x00\x00\x00\x20ftypisom') or content.startswith(b'\x00\x00\x00\x20ftypmp4'):  # MP4
            return ModalityType.VIDEO
        elif content.startswith(b'RIFF') and b'AVI ' in content[:12]:  # AVI
            return ModalityType.VIDEO
        elif content.startswith(b'\x1A\x45\xDF\xA3'):  # MKV/WebM
            return ModalityType.VIDEO
        elif content.startswith(b'FLV'):  # FLV
            return ModalityType.VIDEO

        # Documents and other formats
        elif content.startswith(b'%PDF'):
            return ModalityType.MIXED

        # Try to decode as text
        try:
            content.decode('utf-8')
            return ModalityType.TEXT
        except UnicodeDecodeError:
            return ModalityType.MIXED

    def _get_text_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations for text content."""
        structure = analysis.get('structure', {})
        complexity = analysis.get('complexity', {})
        language = analysis.get('language', 'en')

        recommendations = {
            'primary_strategies': [],
            'secondary_strategies': [],
            'parameters': {},
            'considerations': []
        }

        structure_type = structure.get('structure_type', 'mixed')
        complexity_score = complexity.get('complexity_score', 0.5)

        if structure_type == 'structured_document':
            recommendations['primary_strategies'] = ['section_based', 'paragraph_based', 'semantic_basic']
            recommendations['parameters']['respect_structure'] = True
        elif structure_type == 'technical_document':
            recommendations['primary_strategies'] = ['code_generic', 'syntactic', 'structure_aware']
            recommendations['parameters']['preserve_code_blocks'] = True
        elif structure_type == 'narrative':
            recommendations['primary_strategies'] = ['paragraph_based', 'sentence_based', 'semantic_basic']
            recommendations['parameters']['maintain_flow'] = True
        else:
            recommendations['primary_strategies'] = ['sentence_based', 'paragraph_based']

        # Add complexity-based recommendations
        if complexity_score > 0.7:
            recommendations['considerations'].append('High complexity text - consider semantic chunking')
            recommendations['parameters']['chunk_size'] = 'large'
        elif complexity_score < 0.3:
            recommendations['considerations'].append('Simple text - smaller chunks may work well')
            recommendations['parameters']['chunk_size'] = 'small'

        # Language-specific recommendations
        lang_suggestions = self.language_detector.get_chunking_suggestions(language)
        recommendations['secondary_strategies'].extend(lang_suggestions.get('sentence_strategies', []))
        recommendations['considerations'].extend(lang_suggestions.get('special_considerations', []))

        return recommendations

    def _get_image_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations for image content."""
        return {
            'primary_strategies': ['tile_based', 'patch_based', 'region_based'],
            'secondary_strategies': ['image_block_based', 'quadtree'],
            'parameters': {'preserve_spatial_relations': True},
            'considerations': ['Consider image dimensions and resolution']
        }

    def _get_audio_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations for audio content."""
        return {
            'primary_strategies': ['audio_silence_based', 'audio_frame_based'],
            'secondary_strategies': ['audio_sample_based', 'audio_vad'],
            'parameters': {'preserve_temporal_structure': True},
            'considerations': ['Consider sample rate and audio quality']
        }

    def _get_video_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations for video content."""
        return {
            'primary_strategies': ['video_scene_based', 'video_keyframe'],
            'secondary_strategies': ['video_frame_based', 'video_shot_based'],
            'parameters': {'preserve_temporal_continuity': True},
            'considerations': ['Consider frame rate and video length']
        }

    def _get_generic_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get generic recommendations for unknown content."""
        return {
            'primary_strategies': ['fixed_size', 'rolling_hash'],
            'secondary_strategies': ['rabin_cdc', 'fastcdc'],
            'parameters': {'use_generic_approach': True},
            'considerations': ['Content type unknown - using generic strategies']
        }
