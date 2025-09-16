"""
Java code chunking strategy.

Provides intelligent chunking for Java source code that preserves:
- Package declarations and imports
- Class and interface definitions
- Method and constructor boundaries
- Field declarations
- Annotations and JavaDoc
- Inner classes and enums
- Static blocks and initializers
"""

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from chunking_strategy.core.base import BaseChunker, Chunk, ChunkingResult, ChunkMetadata, ModalityType
from chunking_strategy.core.registry import register_chunker, ComplexityLevel, SpeedLevel, MemoryUsage
from chunking_strategy.core.streaming import StreamableChunker
from chunking_strategy.core.adaptive import AdaptableChunker


@register_chunker(
    name="java_code",
    category="code",
    complexity=ComplexityLevel.MEDIUM,
    speed=SpeedLevel.FAST,
    memory=MemoryUsage.LOW,
    supported_formats=["java"],
    dependencies=[],
    description="Intelligent Java code chunking preserving class, method, and package boundaries",
    use_cases=["code_analysis", "documentation_generation", "code_search", "enterprise_analysis"]
)
class JavaChunker(StreamableChunker, AdaptableChunker):
    """
    Intelligent Java code chunker that preserves syntactic boundaries.
    
    Features:
    - Package declaration and import grouping
    - Class and interface boundary preservation
    - Method and constructor extraction
    - Field and property detection
    - Annotation processing
    - Inner class and enum handling
    - JavaDoc comment preservation
    """

    def __init__(
        self,
        chunk_by: str = "method",  # "method", "class", "interface", "field", "logical_block", "line_count"
        max_lines_per_chunk: int = 150,
        include_package: bool = True,
        include_imports: bool = True,
        include_annotations: bool = True,
        include_javadoc: bool = True,
        group_inner_classes: bool = True,
        preserve_access_modifiers: bool = True,
        **kwargs
    ):
        """
        Initialize Java code chunker.

        Args:
            chunk_by: Chunking granularity ("method", "class", "interface", "field", "logical_block", "line_count")
            max_lines_per_chunk: Maximum lines per chunk for line_count mode
            include_package: Whether to include package declaration
            include_imports: Whether to include import statements
            include_annotations: Whether to include annotations
            include_javadoc: Whether to include JavaDoc comments
            group_inner_classes: Whether to group inner classes with outer classes
            preserve_access_modifiers: Whether to preserve access modifier information
            **kwargs: Additional parameters
        """
        super().__init__(
            name="java_code",
            category="code",
            supported_modalities=[ModalityType.TEXT],
            **kwargs
        )
        
        self.chunk_by = chunk_by
        self.max_lines_per_chunk = max_lines_per_chunk
        self.include_package = include_package
        self.include_imports = include_imports
        self.include_annotations = include_annotations
        self.include_javadoc = include_javadoc
        self.group_inner_classes = group_inner_classes
        self.preserve_access_modifiers = preserve_access_modifiers
        
        self.logger = logging.getLogger(__name__)
        self._adaptation_history = []

    def chunk(
        self,
        content: Union[str, bytes, Path],
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk Java code content preserving syntactic boundaries.

        Args:
            content: Java code content or file path
            source_info: Information about the content source
            **kwargs: Additional chunking parameters

        Returns:
            ChunkingResult with Java code chunks
        """
        start_time = time.time()
        
        # Handle input types
        if isinstance(content, Path):
            source_path = content
            with open(content, 'r', encoding='utf-8') as f:
                code_content = f.read()
        elif isinstance(content, str) and len(content) > 0 and len(content) < 500 and '\n' not in content:
            try:
                if Path(content).exists() and Path(content).is_file():
                    source_path = Path(content)
                    with open(source_path, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                else:
                    source_path = None
                    code_content = str(content)
            except (OSError, ValueError):
                source_path = None
                code_content = str(content)
        else:
            source_path = None
            code_content = str(content)

        try:
            # Parse Java code
            lines = code_content.split('\n')
            java_elements = self._extract_java_elements(lines)
            
            # Create chunks based on strategy
            chunks = self._create_chunks_from_elements(
                java_elements, 
                lines, 
                source_path or "direct_input"
            )
            
            # Create chunking result
            result = ChunkingResult(
                chunks=chunks,
                processing_time=time.time() - start_time,
                strategy_used="java_code",
                source_info={
                    "source_file": str(source_path) if source_path else "direct_input",
                    "chunk_by": self.chunk_by,
                    "total_elements": len(java_elements),
                    "total_lines": len(lines),
                    "package_name": self._extract_package_name(lines),
                    "class_names": self._extract_class_names(java_elements)
                }
            )
            
            self.logger.info(f"Java code chunking completed: {len(chunks)} chunks from {source_path or 'direct input'}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing Java code: {e}")
            # Fallback to line-based chunking
            return self._fallback_line_chunking(code_content, source_path, start_time)

    def _extract_package_name(self, lines: List[str]) -> Optional[str]:
        """Extract package name from Java file."""
        for line in lines[:20]:  # Package should be near the top
            line = line.strip()
            if line.startswith('package '):
                return line.split()[1].rstrip(';')
        return None

    def _extract_class_names(self, elements: List[Dict[str, Any]]) -> List[str]:
        """Extract class names from parsed elements."""
        class_names = []
        for element in elements:
            if element['type'] in ['class', 'interface', 'enum']:
                class_names.append(element.get('name', 'unnamed'))
        return class_names

    def _extract_java_elements(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract Java code elements."""
        elements = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Extract JavaDoc comments
            if line.startswith('/**') and self.include_javadoc:
                element = self._extract_javadoc(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno']
                else:
                    i += 1
                continue
            
            # Skip single-line comments
            if line.startswith('//') or line.startswith('/*'):
                i += 1
                continue
            
            # Extract package declaration
            if line.startswith('package ') and self.include_package:
                element = self._extract_package(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno']
                else:
                    i += 1
                continue
            
            # Extract import statements
            if line.startswith('import ') and self.include_imports:
                element = self._extract_import(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno']
                else:
                    i += 1
                continue
            
            # Extract annotations
            if line.startswith('@') and self.include_annotations:
                element = self._extract_annotation(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno']
                else:
                    i += 1
                continue
            
            # Extract class definitions
            if self._is_class_start(line):
                element = self._extract_class(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno'] + 1
                else:
                    i += 1
                continue
            
            # Extract interface definitions
            if self._is_interface_start(line):
                element = self._extract_interface(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno'] + 1
                else:
                    i += 1
                continue
            
            # Extract enum definitions
            if self._is_enum_start(line):
                element = self._extract_enum(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno'] + 1
                else:
                    i += 1
                continue
            
            # Extract method definitions
            if self._is_method_start(line, lines, i):
                element = self._extract_method(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno'] + 1
                else:
                    i += 1
                continue
            
            # Extract field declarations
            if self._is_field_declaration(line):
                element = self._extract_field(lines, i)
                if element:
                    elements.append(element)
                    i = element['end_lineno']
                else:
                    i += 1
                continue
            
            i += 1
        
        return elements

    def _extract_package(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract package declaration."""
        line = lines[start_idx].strip()
        package_match = re.match(r'package\s+([\w.]+)\s*;', line)
        
        if package_match:
            package_name = package_match.group(1)
            return {
                'type': 'package',
                'name': package_name,
                'content': line,
                'lineno': start_idx + 1,
                'end_lineno': start_idx + 1
            }
        return None

    def _extract_import(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract import statement."""
        line = lines[start_idx].strip()
        
        # Handle import statement
        import_match = re.match(r'import\s+(static\s+)?([\w.*]+)\s*;', line)
        if import_match:
            is_static = bool(import_match.group(1))
            import_name = import_match.group(2)
            
            return {
                'type': 'import',
                'name': import_name,
                'content': line,
                'lineno': start_idx + 1,
                'end_lineno': start_idx + 1,
                'is_static': is_static,
                'is_wildcard': import_name.endswith('.*')
            }
        
        return None

    def _extract_javadoc(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract JavaDoc comment block."""
        javadoc_lines = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i]
            javadoc_lines.append(line)
            
            if '*/' in line:
                content = '\n'.join(javadoc_lines)
                
                # Extract JavaDoc tags
                tags = self._parse_javadoc_tags(content)
                
                return {
                    'type': 'javadoc',
                    'name': 'javadoc_comment',
                    'content': content,
                    'lineno': start_idx + 1,
                    'end_lineno': i + 1,
                    'tags': tags
                }
            i += 1
        
        return None

    def _parse_javadoc_tags(self, content: str) -> Dict[str, List[str]]:
        """Parse JavaDoc tags from comment content."""
        tags = {}
        
        # Common JavaDoc tags
        tag_patterns = {
            'param': r'@param\s+(\w+)\s+(.*)',
            'return': r'@return\s+(.*)',
            'throws': r'@throws\s+(\w+)\s*(.*)',
            'see': r'@see\s+(.*)',
            'since': r'@since\s+(.*)',
            'author': r'@author\s+(.*)',
            'version': r'@version\s+(.*)',
            'deprecated': r'@deprecated\s*(.*)'
        }
        
        for tag_name, pattern in tag_patterns.items():
            matches = re.findall(pattern, content, re.MULTILINE)
            if matches:
                tags[tag_name] = matches
        
        return tags

    def _extract_annotation(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract annotation."""
        line = lines[start_idx].strip()
        
        # Simple annotation: @Override
        if re.match(r'^@\w+$', line):
            annotation_name = line[1:]  # Remove @
            return {
                'type': 'annotation',
                'name': annotation_name,
                'content': line,
                'lineno': start_idx + 1,
                'end_lineno': start_idx + 1,
                'has_parameters': False
            }
        
        # Annotation with parameters: @SuppressWarnings("unchecked")
        annotation_match = re.match(r'^@(\w+)\s*\((.*)\)$', line)
        if annotation_match:
            annotation_name = annotation_match.group(1)
            parameters = annotation_match.group(2)
            
            return {
                'type': 'annotation',
                'name': annotation_name,
                'content': line,
                'lineno': start_idx + 1,
                'end_lineno': start_idx + 1,
                'has_parameters': True,
                'parameters': parameters
            }
        
        return None

    def _is_class_start(self, line: str) -> bool:
        """Check if line starts a class definition."""
        # Class pattern: [modifiers] class ClassName [extends] [implements] {
        class_patterns = [
            r'\bclass\s+\w+',
            r'(public|private|protected|abstract|final|static)\s+.*\bclass\s+\w+',
        ]
        
        for pattern in class_patterns:
            if re.search(pattern, line):
                return True
        return False

    def _extract_class(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract class definition."""
        first_line = lines[start_idx].strip()
        
        # Parse class signature
        class_info = self._parse_class_signature(first_line)
        if not class_info:
            return None
        
        # Find class body end
        brace_count = 0
        class_lines = []
        in_class = False
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            class_lines.append(line)
            
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_class = True
                elif char == '}':
                    brace_count -= 1
                    
                    if in_class and brace_count == 0:
                        content = '\n'.join(class_lines)
                        
                        # Extract class members
                        members = self._extract_class_members(class_lines)
                        
                        return {
                            'type': 'class',
                            'name': class_info['name'],
                            'content': content,
                            'lineno': start_idx + 1,
                            'end_lineno': i + 1,
                            'access_modifier': class_info['access_modifier'],
                            'modifiers': class_info['modifiers'],
                            'extends': class_info.get('extends'),
                            'implements': class_info.get('implements', []),
                            'members': members
                        }
        
        return None

    def _parse_class_signature(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse class signature to extract details."""
        # Extract class name
        class_match = re.search(r'\bclass\s+(\w+)', line)
        if not class_match:
            return None
        
        class_name = class_match.group(1)
        
        # Extract access modifier
        access_modifier = 'default'
        if 'public' in line:
            access_modifier = 'public'
        elif 'private' in line:
            access_modifier = 'private'
        elif 'protected' in line:
            access_modifier = 'protected'
        
        # Extract modifiers
        modifiers = []
        modifier_patterns = ['abstract', 'final', 'static']
        for modifier in modifier_patterns:
            if f' {modifier} ' in f' {line} ':
                modifiers.append(modifier)
        
        # Extract extends
        extends_match = re.search(r'\bextends\s+(\w+)', line)
        extends = extends_match.group(1) if extends_match else None
        
        # Extract implements
        implements_match = re.search(r'\bimplements\s+([^{]+)', line)
        implements = []
        if implements_match:
            implements_str = implements_match.group(1).strip()
            implements = [iface.strip() for iface in implements_str.split(',')]
        
        return {
            'name': class_name,
            'access_modifier': access_modifier,
            'modifiers': modifiers,
            'extends': extends,
            'implements': implements
        }

    def _extract_class_members(self, class_lines: List[str]) -> Dict[str, List[str]]:
        """Extract class members (methods, fields, etc.)."""
        members = {
            'methods': [],
            'fields': [],
            'constructors': [],
            'inner_classes': []
        }
        
        # Simple member extraction - could be enhanced
        for line in class_lines:
            line = line.strip()
            
            # Method detection
            if ('(' in line and ')' in line and '{' in line and 
                not line.startswith('//') and not line.startswith('/*')):
                method_match = re.search(r'(\w+)\s*\([^)]*\)\s*{', line)
                if method_match:
                    members['methods'].append(method_match.group(1))
            
            # Field detection
            elif (';' in line and '=' in line and 
                  not line.startswith('//') and not line.startswith('/*')):
                field_match = re.search(r'(\w+)\s*[=;]', line)
                if field_match:
                    members['fields'].append(field_match.group(1))
        
        return members

    def _is_interface_start(self, line: str) -> bool:
        """Check if line starts an interface definition."""
        return bool(re.search(r'\binterface\s+\w+', line))

    def _extract_interface(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract interface definition."""
        first_line = lines[start_idx].strip()
        
        # Extract interface name
        interface_match = re.search(r'\binterface\s+(\w+)', first_line)
        if not interface_match:
            return None
        
        interface_name = interface_match.group(1)
        
        # Find interface end
        brace_count = 0
        interface_lines = []
        in_interface = False
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            interface_lines.append(line)
            
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_interface = True
                elif char == '}':
                    brace_count -= 1
                    
                    if in_interface and brace_count == 0:
                        content = '\n'.join(interface_lines)
                        
                        # Extract interface methods
                        methods = self._extract_interface_methods(interface_lines)
                        
                        return {
                            'type': 'interface',
                            'name': interface_name,
                            'content': content,
                            'lineno': start_idx + 1,
                            'end_lineno': i + 1,
                            'methods': methods
                        }
        
        return None

    def _extract_interface_methods(self, interface_lines: List[str]) -> List[str]:
        """Extract method signatures from interface definition."""
        methods = []
        
        for line in interface_lines:
            line = line.strip()
            # Interface method: return_type method_name(params);
            if ('(' in line and ')' in line and ';' in line and 
                not line.startswith('//') and not line.startswith('/*')):
                method_match = re.search(r'(\w+)\s*\([^)]*\)\s*;', line)
                if method_match:
                    methods.append(method_match.group(1))
        
        return methods

    def _is_enum_start(self, line: str) -> bool:
        """Check if line starts an enum definition."""
        return bool(re.search(r'\benum\s+\w+', line))

    def _extract_enum(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract enum definition."""
        first_line = lines[start_idx].strip()
        
        # Extract enum name
        enum_match = re.search(r'\benum\s+(\w+)', first_line)
        if not enum_match:
            return None
        
        enum_name = enum_match.group(1)
        
        # Find enum end
        brace_count = 0
        enum_lines = []
        in_enum = False
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            enum_lines.append(line)
            
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_enum = True
                elif char == '}':
                    brace_count -= 1
                    
                    if in_enum and brace_count == 0:
                        content = '\n'.join(enum_lines)
                        
                        # Extract enum constants
                        constants = self._extract_enum_constants(enum_lines)
                        
                        return {
                            'type': 'enum',
                            'name': enum_name,
                            'content': content,
                            'lineno': start_idx + 1,
                            'end_lineno': i + 1,
                            'constants': constants
                        }
        
        return None

    def _extract_enum_constants(self, enum_lines: List[str]) -> List[str]:
        """Extract enum constants."""
        constants = []
        
        for line in enum_lines[1:-1]:  # Skip opening and closing braces
            line = line.strip()
            if line and not line.startswith('//') and not line.startswith('/*'):
                # Enum constant: CONSTANT_NAME,
                constant_match = re.match(r'^([A-Z_][A-Z0-9_]*)', line)
                if constant_match:
                    constants.append(constant_match.group(1))
        
        return constants

    def _is_method_start(self, line: str, lines: List[str], idx: int) -> bool:
        """Check if line starts a method definition."""
        # Method patterns
        patterns = [
            r'\b\w+\s+\w+\s*\([^)]*\)\s*{',  # return_type method_name(params) {
            r'\b(public|private|protected)\s+.*\w+\s*\([^)]*\)\s*{',  # access modifier
            r'\b(static|final|abstract)\s+.*\w+\s*\([^)]*\)\s*{',  # other modifiers
        ]
        
        # Skip lines that look like class declarations
        if 'class ' in line or 'interface ' in line or 'enum ' in line:
            return False
        
        for pattern in patterns:
            if re.search(pattern, line):
                return True
        
        return False

    def _extract_method(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract method definition."""
        first_line = lines[start_idx].strip()
        
        # Parse method signature
        method_info = self._parse_method_signature(first_line)
        if not method_info:
            return None
        
        # Find method body end
        brace_count = 0
        method_lines = []
        in_method = False
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            method_lines.append(line)
            
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_method = True
                elif char == '}':
                    brace_count -= 1
                    
                    if in_method and brace_count == 0:
                        content = '\n'.join(method_lines)
                        
                        return {
                            'type': 'method',
                            'name': method_info['name'],
                            'content': content,
                            'lineno': start_idx + 1,
                            'end_lineno': i + 1,
                            'access_modifier': method_info['access_modifier'],
                            'modifiers': method_info['modifiers'],
                            'return_type': method_info['return_type'],
                            'parameters': method_info['parameters'],
                            'is_constructor': method_info['is_constructor']
                        }
        
        return None

    def _parse_method_signature(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse method signature to extract details."""
        # Extract method name and parameters
        method_match = re.search(r'(\w+)\s*\(([^)]*)\)', line)
        if not method_match:
            return None
        
        method_name = method_match.group(1)
        params_str = method_match.group(2).strip()
        
        # Extract access modifier
        access_modifier = 'default'
        if 'public' in line:
            access_modifier = 'public'
        elif 'private' in line:
            access_modifier = 'private'
        elif 'protected' in line:
            access_modifier = 'protected'
        
        # Extract modifiers
        modifiers = []
        modifier_patterns = ['static', 'final', 'abstract', 'synchronized', 'native']
        for modifier in modifier_patterns:
            if f' {modifier} ' in f' {line} ':
                modifiers.append(modifier)
        
        # Extract return type (approximate)
        parts = line.split()
        return_type = 'void'
        for i, part in enumerate(parts):
            if part == method_name and i > 0:
                return_type = parts[i-1]
                break
        
        # Parse parameters
        parameters = []
        if params_str:
            for param in params_str.split(','):
                param = param.strip()
                if param:
                    param_parts = param.split()
                    if len(param_parts) >= 2:
                        param_type = param_parts[-2]
                        param_name = param_parts[-1]
                        parameters.append(f"{param_type} {param_name}")
        
        # Check if it's a constructor (return type same as method name or no return type before method name)
        is_constructor = return_type == method_name or return_type in ['public', 'private', 'protected']
        
        return {
            'name': method_name,
            'access_modifier': access_modifier,
            'modifiers': modifiers,
            'return_type': return_type if not is_constructor else None,
            'parameters': parameters,
            'is_constructor': is_constructor
        }

    def _is_field_declaration(self, line: str) -> bool:
        """Check if line is a field declaration."""
        # Field declaration ends with semicolon and doesn't contain method patterns
        return (line.endswith(';') and 
                '(' not in line and 
                not line.startswith('//') and 
                not line.startswith('/*') and
                not line.startswith('@') and
                not 'class ' in line and
                not 'interface ' in line and
                not 'enum ' in line)

    def _extract_field(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Extract field declaration."""
        line = lines[start_idx].strip()
        
        # Parse field declaration
        # Example: private static final String CONSTANT = "value";
        field_match = re.search(r'(\w+)\s*[=;]', line)
        if not field_match:
            return None
        
        field_name = field_match.group(1)
        
        # Extract access modifier
        access_modifier = 'default'
        if 'public' in line:
            access_modifier = 'public'
        elif 'private' in line:
            access_modifier = 'private'
        elif 'protected' in line:
            access_modifier = 'protected'
        
        # Extract modifiers
        modifiers = []
        modifier_patterns = ['static', 'final', 'volatile', 'transient']
        for modifier in modifier_patterns:
            if f' {modifier} ' in f' {line} ':
                modifiers.append(modifier)
        
        # Extract field type (approximate)
        parts = line.split()
        field_type = 'Object'
        for i, part in enumerate(parts):
            if part == field_name and i > 0:
                field_type = parts[i-1]
                break
        
        return {
            'type': 'field',
            'name': field_name,
            'content': line,
            'lineno': start_idx + 1,
            'end_lineno': start_idx + 1,
            'access_modifier': access_modifier,
            'modifiers': modifiers,
            'field_type': field_type,
            'has_initializer': '=' in line
        }

    def _create_chunks_from_elements(
        self, 
        elements: List[Dict[str, Any]], 
        lines: List[str], 
        source: str
    ) -> List[Chunk]:
        """Create chunks from extracted Java elements."""
        chunks = []
        
        if self.chunk_by == "method":
            # Create one chunk per method
            for element in elements:
                if element['type'] in ['method', 'package', 'import', 'javadoc', 'annotation']:
                    chunk = self._create_chunk_from_element(element, source)
                    chunks.append(chunk)
        
        elif self.chunk_by == "class":
            # Create one chunk per class
            for element in elements:
                if element['type'] in ['class', 'interface', 'enum', 'package', 'import']:
                    chunk = self._create_chunk_from_element(element, source)
                    chunks.append(chunk)
        
        elif self.chunk_by == "interface":
            # Create one chunk per interface
            for element in elements:
                if element['type'] in ['interface', 'package', 'import']:
                    chunk = self._create_chunk_from_element(element, source)
                    chunks.append(chunk)
        
        elif self.chunk_by == "field":
            # Create one chunk per field
            for element in elements:
                if element['type'] in ['field', 'package', 'import']:
                    chunk = self._create_chunk_from_element(element, source)
                    chunks.append(chunk)
        
        elif self.chunk_by == "logical_block":
            # Group related elements together
            chunks = self._create_logical_blocks(elements, source)
        
        elif self.chunk_by == "line_count":
            # Chunk by line count while respecting boundaries
            chunks = self._create_line_count_chunks(elements, lines, source)
        
        return chunks

    def _create_chunk_from_element(self, element: Dict[str, Any], source: str) -> Chunk:
        """Create a chunk from a single Java element."""
        chunk_metadata = ChunkMetadata(
            source=source,
            position={"start_line": element['lineno'], "end_line": element['end_lineno']},
            chunker_used="java_code",
            extra={
                "chunk_type": "code",
                "language": "java",
                "element_type": element['type'],
                "element_name": element.get('name', 'unnamed'),
                "access_modifier": element.get('access_modifier'),
                "modifiers": element.get('modifiers', []),
                "return_type": element.get('return_type'),
                "field_type": element.get('field_type'),
                "parameters": element.get('parameters', []),
                "extends": element.get('extends'),
                "implements": element.get('implements', []),
                "members": element.get('members', {}),
                "methods": element.get('methods', []),
                "fields": element.get('fields', []),
                "constants": element.get('constants', []),
                "is_constructor": element.get('is_constructor', False),
                "is_static": element.get('is_static', False),
                "has_initializer": element.get('has_initializer', False),
                "has_parameters": element.get('has_parameters', False),
                "javadoc_tags": element.get('tags', {})
            }
        )
        
        chunk_id = f"java_{element['type']}_{element.get('name', 'unnamed')}_{element['lineno']}"
        
        return Chunk(
            id=chunk_id,
            content=element['content'],
            modality=ModalityType.TEXT,
            metadata=chunk_metadata
        )

    def _create_logical_blocks(self, elements: List[Dict[str, Any]], source: str) -> List[Chunk]:
        """Create chunks by grouping logically related elements."""
        chunks = []
        current_block = []
        current_lines = 0
        
        for element in elements:
            element_lines = element['end_lineno'] - element['lineno'] + 1
            
            if current_lines + element_lines > self.max_lines_per_chunk and current_block:
                chunk = self._create_block_chunk(current_block, source, len(chunks))
                chunks.append(chunk)
                current_block = [element]
                current_lines = element_lines
            else:
                current_block.append(element)
                current_lines += element_lines
        
        if current_block:
            chunk = self._create_block_chunk(current_block, source, len(chunks))
            chunks.append(chunk)
        
        return chunks

    def _create_block_chunk(self, elements: List[Dict[str, Any]], source: str, chunk_index: int) -> Chunk:
        """Create a chunk from multiple elements."""
        content = '\n\n'.join(element['content'] for element in elements)
        
        start_line = min(element['lineno'] for element in elements)
        end_line = max(element['end_lineno'] for element in elements)
        
        element_names = [element.get('name', 'unnamed') for element in elements]
        element_types = [element['type'] for element in elements]
        
        chunk_metadata = ChunkMetadata(
            source=source,
            position={"start_line": start_line, "end_line": end_line},
            chunker_used="java_code",
            extra={
                "chunk_type": "code",
                "language": "java",
                "element_types": element_types,
                "element_names": element_names,
                "block_size": len(elements)
            }
        )
        
        return Chunk(
            id=f"java_block_{chunk_index}_{start_line}",
            content=content,
            modality=ModalityType.TEXT,
            metadata=chunk_metadata
        )

    def _create_line_count_chunks(
        self, 
        elements: List[Dict[str, Any]], 
        lines: List[str], 
        source: str
    ) -> List[Chunk]:
        """Create chunks based on line count while respecting element boundaries."""
        chunks = []
        current_lines = []
        current_elements = []
        
        for element in elements:
            element_lines = element['end_lineno'] - element['lineno'] + 1
            
            if len(current_lines) + element_lines > self.max_lines_per_chunk and current_lines:
                chunk = self._create_line_chunk(current_lines, current_elements, source, len(chunks))
                chunks.append(chunk)
                current_lines = []
                current_elements = []
            
            element_content_lines = element['content'].split('\n')
            current_lines.extend(element_content_lines)
            current_elements.append(element)
        
        if current_lines:
            chunk = self._create_line_chunk(current_lines, current_elements, source, len(chunks))
            chunks.append(chunk)
        
        return chunks

    def _create_line_chunk(
        self, 
        lines: List[str], 
        elements: List[Dict[str, Any]], 
        source: str, 
        chunk_index: int
    ) -> Chunk:
        """Create a chunk from lines and associated elements."""
        content = '\n'.join(lines)
        
        start_line = min(element['lineno'] for element in elements) if elements else 1
        end_line = max(element['end_lineno'] for element in elements) if elements else len(lines)
        
        chunk_metadata = ChunkMetadata(
            source=source,
            position={"start_line": start_line, "end_line": end_line},
            chunker_used="java_code",
            extra={
                "chunk_type": "code",
                "language": "java",
                "line_count": len(lines),
                "elements_included": len(elements)
            }
        )
        
        return Chunk(
            id=f"java_lines_{chunk_index}_{start_line}",
            content=content,
            modality=ModalityType.TEXT,
            metadata=chunk_metadata
        )

    def _fallback_line_chunking(self, content: str, source_path: Optional[Path], start_time: float) -> ChunkingResult:
        """Fallback to simple line-based chunking."""
        lines = content.split('\n')
        chunks = []
        
        for i in range(0, len(lines), self.max_lines_per_chunk):
            chunk_lines = lines[i:i + self.max_lines_per_chunk]
            chunk_content = '\n'.join(chunk_lines)
            
            chunk = Chunk(
                id=f"java_fallback_{i // self.max_lines_per_chunk}",
                content=chunk_content,
                modality=ModalityType.TEXT,
                metadata=ChunkMetadata(
                    source=str(source_path) if source_path else "direct_input",
                    position={"start_line": i + 1, "end_line": min(i + len(chunk_lines), len(lines))},
                    chunker_used="java_code",
                    extra={
                        "chunk_type": "code",
                        "language": "java",
                        "fallback": True
                    }
                )
            )
            chunks.append(chunk)
        
        return ChunkingResult(
            chunks=chunks,
            processing_time=time.time() - start_time,
            strategy_used="java_code_fallback"
        )

    def chunk_stream(self, content_stream, **kwargs):
        """Chunk Java code from a stream."""
        # Collect stream content
        content_lines = []
        for chunk in content_stream:
            if isinstance(chunk, str):
                content_lines.extend(chunk.split('\n'))
            elif isinstance(chunk, bytes):
                content_lines.extend(chunk.decode('utf-8').split('\n'))
        
        content = '\n'.join(content_lines)
        result = self.chunk(content, **kwargs)
        
        for chunk in result.chunks:
            yield chunk

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return ["java"]

    def estimate_chunks(self, content: Union[str, Path]) -> int:
        """Estimate number of chunks that will be generated."""
        if isinstance(content, (str, Path)):
            if isinstance(content, Path) and content.exists():
                with open(content, 'r') as f:
                    lines = len(f.readlines())
            else:
                lines = len(str(content).split('\n'))
            
            if self.chunk_by in ["method", "class", "interface", "field"]:
                return max(1, lines // 25)  # Estimate methods/classes
            else:
                return max(1, lines // self.max_lines_per_chunk)
        
        return 1

    def adapt_parameters(
        self,
        feedback_score: float,
        feedback_type: str = "quality",
        **kwargs
    ) -> None:
        """Adapt chunker parameters based on feedback."""
        adaptation = {
            "timestamp": time.time(),
            "feedback_score": feedback_score,
            "feedback_type": feedback_type,
            "old_max_lines": self.max_lines_per_chunk
        }
        
        if feedback_type == "quality" and feedback_score < 0.5:
            old_max = self.max_lines_per_chunk
            self.max_lines_per_chunk = max(30, int(self.max_lines_per_chunk * 0.8))
            adaptation["new_max_lines"] = self.max_lines_per_chunk
            self.logger.info(f"Adapted max_lines_per_chunk: {old_max} -> {self.max_lines_per_chunk}")
        elif feedback_type == "performance" and feedback_score < 0.5:
            old_max = self.max_lines_per_chunk
            self.max_lines_per_chunk = min(500, int(self.max_lines_per_chunk * 1.2))
            adaptation["new_max_lines"] = self.max_lines_per_chunk
            self.logger.info(f"Adapted max_lines_per_chunk: {old_max} -> {self.max_lines_per_chunk}")
        
        self._adaptation_history.append(adaptation)

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter adaptations."""
        return self._adaptation_history.copy()
