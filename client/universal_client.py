"""
Universal Client for Subscription Analytics
- Command-line and interactive analytics using natural language
- Graph generation and feedback-driven improvement
- Connects to remote API and Gemini AI
"""

# client/universal_client.py - COMPLETE FIXED VERSION WITH ALL FUNCTIONALITY AND MULTITOOL SUPPORT

import asyncio
import aiohttp
import os
import json
import sys
import ssl
import certifi
import logging
import re
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import google.generativeai as genai
from datetime import datetime, timedelta
import argparse
import calendar

# Graph visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Matplotlib available for graph generation")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ Matplotlib not available - graphs will be disabled")

from pathlib import Path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Try to import config manager with multiple fallback paths
try:
    from config_manager import ConfigManager
except ImportError:
    try:
        # Try importing from parent directory
        sys.path.insert(0, str(current_dir.parent))
        from config_manager import ConfigManager
    except ImportError:
        # Create a minimal config manager if not found
        class ConfigManager:
            def __init__(self):
                self.config_path = Path.cwd() / 'config.json'
            
            def get_config(self):
                if self.config_path.exists():
                    with open(self.config_path, 'r') as f:
                        return json.load(f)
                return {}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
GREY = "\033[90m"

def color_text(text, color):
    return f"{color}{text}{RESET}"

def print_header(text):
    print(f"{BOLD}{CYAN}\n{'='*len(text)}\n{text}\n{'='*len(text)}{RESET}")

def print_section(text):
    print(f"{BOLD}{BLUE}\n{text}{RESET}")

def print_separator():
    print(f"{GREY}{'-'*60}{RESET}")

def print_success(text):
    print(f"{GREEN}{text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}{text}{RESET}")

def print_error(text):
    print(f"{RED}{text}{RESET}")

def print_feedback_prompt(text):
    print(f"{MAGENTA}{text}{RESET}")

@dataclass
class QueryResult:
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    tool_used: Optional[str] = None
    parameters: Optional[Dict] = None
    is_dynamic: bool = False
    original_query: Optional[str] = None
    generated_sql: Optional[str] = None
    message: Optional[str] = None
    graph_data: Optional[Dict] = None
    graph_generated: bool = False

class CompleteGraphGenerator:
    """COMPLETE graph generator with full smart data handling and production-ready features."""
    
    def __init__(self):
        self.graphs_dir = self._setup_graphs_directory()
        self.supported_types = ['line', 'bar', 'horizontal_bar', 'pie', 'scatter']
        
    def _setup_graphs_directory(self) -> Path:
        """Setup graphs directory with smart fallbacks."""
        possible_dirs = [
            Path.cwd() / "generated_graphs",
            Path(__file__).parent / "generated_graphs",
            Path.home() / "subscription_graphs",
            Path.cwd()  # Final fallback
        ]
        
        for directory in possible_dirs:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                # Test write permissions
                test_file = directory / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
                logger.info(f"📊 Graph directory set to: {directory}")
                return directory
            except Exception as e:
                logger.debug(f"Cannot use directory {directory}: {e}")
                continue
        
        raise RuntimeError("No writable directory found for graph generation")
    
    def can_generate_graphs(self) -> bool:
        return MATPLOTLIB_AVAILABLE
    
    def generate_graph(self, graph_data: Dict, query: str) -> Optional[str]:
        """Generate graph with complete enhanced error handling and smart type enforcement."""
        if not self.can_generate_graphs():
            logger.warning("Cannot generate graph - matplotlib not available")
            return None
        try:
            # FIXED: Better data validation and preparation
            if not self._validate_and_prepare_graph_data(graph_data):
                logger.error("Invalid graph data structure after preparation")
                return None
            
            # ENFORCE: Use the graph_type from graph_data (set by tool call override logic)
            requested_graph_type = graph_data.get('graph_type', '').lower()
            query_lower = query.lower()
            logger.info(f"[GRAPH] Received graph_type: '{requested_graph_type}' from tool call. Query: {query}")
            
            # Only use requested_graph_type if valid, else fallback to smart detection
            if requested_graph_type in self.supported_types:
                graph_type = requested_graph_type
            else:
                # Smart fallback if not set or invalid
                graph_type = self._determine_optimal_graph_type(graph_data, query)
                logger.info(f"[GRAPH] Falling back to smart-detected graph_type: '{graph_type}'")
            
            logger.info(f"[GRAPH] Actually using graph_type: '{graph_type}' for plotting.")
            
            # Never use pie chart for time series data
            if graph_type == 'pie' and self._is_time_series_data(graph_data):
                logger.warning("Pie chart requested for time series data; switching to line chart.")
                graph_type = 'line'
            
            # Set up matplotlib with error handling
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Generate graph based on type
            success = self._create_graph_by_type(ax, graph_data, graph_type)
            if not success:
                logger.error(f"Failed to create {graph_type} chart")
                plt.close(fig)
                print(f"❌ Could not generate a {graph_type} chart for this data. Try a different chart type or aggregation.")
                return None
            
            # Enhance graph appearance
            self._enhance_graph_appearance(fig, ax, graph_data, graph_type)
            
            # Save with smart naming
            filepath = self._save_graph_safely(fig, graph_type)
            plt.close(fig)
            
            if filepath and filepath.exists():
                self._auto_open_graph(str(filepath))
                logger.info(f"✅ Graph generated successfully: {graph_type}")
                return str(filepath)
            return None
        except Exception as e:
            logger.error(f"Graph generation failed: {e}")
            plt.close('all')
            return None
    
    def _validate_and_prepare_graph_data(self, graph_data: Dict) -> bool:
        """FIXED: Validate and prepare graph data, ensuring proper category/value structure."""
        try:
            # Check if we have raw SQL result data that needs conversion
            if 'data' in graph_data and isinstance(graph_data['data'], list) and len(graph_data['data']) > 0:
                raw_data = graph_data['data']
                if isinstance(raw_data[0], dict):
                    columns = list(raw_data[0].keys())
                    graph_type = graph_data.get('graph_type', '').lower()
                    
                    logger.info(f"[GRAPH] Preparing {graph_type} chart with columns: {columns}")
                    
                    if graph_type == 'pie':
                        # Look for category/value columns for pie charts
                        category_col = None
                        value_col = None
                        
                        # Priority mapping for pie charts
                        for col in columns:
                            col_lower = col.lower()
                            if col_lower in ['category', 'status', 'type', 'label']:
                                category_col = col
                            elif col_lower in ['value', 'count', 'amount', 'total', 'num']:
                                value_col = col
                        
                        # Fallback to first two columns if no exact match
                        if not category_col and len(columns) >= 1:
                            category_col = columns[0]
                        if not value_col and len(columns) >= 2:
                            value_col = columns[1]
                        
                        if category_col and value_col:
                            graph_data['labels'] = [str(row[category_col]) for row in raw_data]
                            graph_data['values'] = [float(row[value_col]) if row[value_col] is not None else 0 for row in raw_data]
                            logger.info(f"[PIE] Prepared data: labels={len(graph_data['labels'])}, values={len(graph_data['values'])}")
                        else:
                            logger.error(f"[PIE] Could not find category/value columns in: {columns}")
                            return False
                    
                    elif graph_type in ['bar', 'horizontal_bar']:
                        # Accept both 'categories'/'values' and 'category'/'value' for bar charts
                        # Try to find the best matching columns
                        category_col = None
                        value_col = None
                        for col in columns:
                            col_lower = col.lower()
                            if col_lower in ['categories', 'category', 'status', 'type', 'label']:
                                category_col = col
                            elif col_lower in ['values', 'value', 'count', 'amount', 'total', 'num']:
                                value_col = col
                        # Fallback to first two columns if no exact match
                        if not category_col and len(columns) >= 1:
                            category_col = columns[0]
                        if not value_col and len(columns) >= 2:
                            value_col = columns[1]
                        # Prepare bar chart data
                        categories = []
                        values = []
                        if category_col and value_col:
                            for row in raw_data:
                                cat_val = row.get(category_col)
                                num_val = row.get(value_col)
                                if cat_val is not None and num_val is not None:
                                    # Truncate long category names for better display
                                    cat_str = str(cat_val)
                                    if len(cat_str) > 15:
                                        cat_str = cat_str[:12] + "..."
                                    categories.append(cat_str)
                                    try:
                                        values.append(float(num_val))
                                    except (ValueError, TypeError):
                                        values.append(0)
                        # If not found, try to use 'categories'/'values' keys directly if present
                        if not categories and not values:
                            if 'categories' in graph_data and 'values' in graph_data:
                                categories = graph_data['categories']
                                values = graph_data['values']
                        # If not found, try to use 'category'/'value' keys directly if present
                        if not categories and not values:
                            if 'category' in graph_data and 'value' in graph_data:
                                categories = graph_data['category']
                                values = graph_data['value']
                        if categories and values and len(categories) == len(values):
                            graph_data['x_values'] = categories
                            graph_data['y_values'] = values
                            graph_data['x_label'] = (category_col or 'Category').replace('_', ' ').title()
                            graph_data['y_label'] = (value_col or 'Value').replace('_', ' ').title()
                            logger.info(f"[BAR] Prepared data: {len(categories)} categories, {len(values)} values")
                            logger.info(f"[BAR] Sample data: {categories[:3]} -> {values[:3]}")
                            return True
                        else:
                            logger.error(f"[BAR] Data preparation failed: categories={len(categories)}, values={len(values)}")
                            return False
                    
                    elif graph_type == 'line':
                        # FIXED: Better handling for line charts
                        if len(columns) >= 2:
                            x_col = columns[0]
                            y_col = columns[1]
                            
                            # Try to find better time/value columns
                            for col in columns:
                                col_lower = col.lower()
                                if any(keyword in col_lower for keyword in ['date', 'time', 'period', 'month', 'year', 'week']):
                                    x_col = col
                                elif any(keyword in col_lower for keyword in ['amount', 'total', 'count', 'value', 'revenue']):
                                    y_col = col
                            
                            x_values = []
                            y_values = []
                            
                            for row in raw_data:
                                x_val = row.get(x_col)
                                y_val = row.get(y_col)
                                
                                if x_val is not None and y_val is not None:
                                    x_values.append(str(x_val))
                                    try:
                                        y_values.append(float(y_val))
                                    except (ValueError, TypeError):
                                        y_values.append(0)
                            
                            # PATCH: Fill missing periods with zero values for time series
                            if x_col and y_col and x_values:
                                # Detect if x_col is week or month period
                                import re
                                week_pattern = re.compile(r'^(\d{4})-W(\d{2})$')
                                month_pattern = re.compile(r'^(\d{4})-(\d{2})$')
                                # Only patch if all x_values match week or month pattern
                                if all(week_pattern.match(x) for x in x_values):
                                    # Fill missing weeks
                                    years_weeks = [tuple(map(int, m.groups())) for x in x_values if (m := week_pattern.match(x))]
                                    min_year, min_week = min(years_weeks)
                                    max_year, max_week = max(years_weeks)
                                    # Build all weeks in range
                                    from datetime import date, timedelta
                                    def week_to_date(year, week):
                                        return date.fromisocalendar(year, week, 1)
                                    all_weeks = set()
                                    d = week_to_date(min_year, min_week)
                                    end = week_to_date(max_year, max_week)
                                    while d <= end:
                                        iso_year, iso_week, _ = d.isocalendar()
                                        all_weeks.add((iso_year, iso_week))
                                        d += timedelta(weeks=1)
                                    # Map x_values to y_values
                                    week_to_value = {yw: y for yw, y in zip(years_weeks, y_values)}
                                    filled_x = []
                                    filled_y = []
                                    for yw in sorted(all_weeks):
                                        label = f"{yw[0]}-W{str(yw[1]).zfill(2)}"
                                        filled_x.append(label)
                                        filled_y.append(week_to_value.get(yw, 0))
                                    x_values = filled_x
                                    y_values = filled_y
                                elif all(month_pattern.match(x) for x in x_values):
                                    # Fill missing months
                                    months = [tuple(map(int, m.groups())) for x in x_values if (m := month_pattern.match(x))]
                                    min_year, min_month = min(months)
                                    max_year, max_month = max(months)
                                    all_months = []
                                    y, m = min_year, min_month
                                    while (y, m) <= (max_year, max_month):
                                        all_months.append((y, m))
                                        if m == 12:
                                            y += 1
                                            m = 1
                                        else:
                                            m += 1
                                    month_to_value = {ym: y for ym, y in zip(months, y_values)}
                                    filled_x = []
                                    filled_y = []
                                    for ym in all_months:
                                        label = f"{ym[0]}-{str(ym[1]).zfill(2)}"
                                        filled_x.append(label)
                                        filled_y.append(month_to_value.get(ym, 0))
                                    x_values = filled_x
                                    y_values = filled_y
                            if x_values and y_values and len(x_values) == len(y_values):
                                graph_data['x_values'] = x_values
                                graph_data['y_values'] = y_values
                                graph_data['x_label'] = x_col.replace('_', ' ').title()
                                graph_data['y_label'] = y_col.replace('_', ' ').title()
                                logger.info(f"[LINE] Prepared data: {len(x_values)} points")
                                return True
                            else:
                                logger.error(f"[LINE] Data preparation failed: x={len(x_values)}, y={len(y_values)}")
                                return False
        
            # Validate final structure based on graph type
            graph_type = graph_data.get('graph_type', '').lower()
            if graph_type == 'pie':
                return 'labels' in graph_data and 'values' in graph_data and len(graph_data['labels']) > 0
            elif graph_type in ['bar', 'horizontal_bar']:
                return ('x_values' in graph_data and 'y_values' in graph_data and 
                       len(graph_data['x_values']) > 0 and len(graph_data['y_values']) > 0)
            elif graph_type == 'line':
                # ADDED: Check for extreme value ranges in line charts
                if 'y_values' in graph_data and len(graph_data['y_values']) > 1:
                    y_values = graph_data['y_values']
                    min_val = min(y_values)
                    max_val = max(y_values)
                    if min_val > 0:
                        ratio = max_val / min_val
                        if ratio > 500:
                            logger.warning(f"[VALIDATION] Extreme value range detected in line chart: {ratio:.1f}")
                            logger.warning(f"[VALIDATION] Small values might be hard to see: min={min_val:,.0f}, max={max_val:,.0f}")
                            graph_data['scaling_warning'] = f"Note: Values range from {min_val:,.0f} to {max_val:,.0f}"
                return ('x_values' in graph_data and 'y_values' in graph_data and 
                       len(graph_data['x_values']) > 0 and len(graph_data['y_values']) > 0)
            return True
            
        except Exception as e:
            logger.error(f"Error in data validation and preparation: {e}")
            return False
    
    def _is_time_series_data(self, graph_data: Dict) -> bool:
        """Check if data represents time series."""
        # Check for time-related column names or x_values
        if 'x_values' in graph_data:
            x_vals = graph_data['x_values']
            if x_vals and isinstance(x_vals[0], str):
                first_val = str(x_vals[0]).lower()
                return any(word in first_val for word in ['january', 'february', 'march', 'april', 'may', 'june', 
                                                         'july', 'august', 'september', 'october', 'november', 'december',
                                                         '2024', '2025', 'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                         'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        
        if 'data' in graph_data and isinstance(graph_data['data'], list) and len(graph_data['data']) > 0:
            columns = list(graph_data['data'][0].keys())
            return any(any(word in col.lower() for word in ['date', 'time', 'period', 'month', 'year']) for col in columns)
        
        return False
    
    def _determine_optimal_graph_type(self, graph_data: Dict, query: str) -> str:
        """Smart graph type determination based on data and query."""
        requested_type = graph_data.get('graph_type', '').lower()
        query_lower = query.lower()
        
        # Check for explicit requests in query
        if any(word in query_lower for word in ['pie chart', 'pie', 'distribution', 'breakdown']):
            return 'pie'
        elif any(word in query_lower for word in ['line chart', 'line', 'trend', 'trends', 'over time', 'timeline', 'payment trends']):
            logger.info(f"[GRAPH] Detected trend/time keywords in query: '{query}' - using line chart")
            return 'line'
        elif any(word in query_lower for word in ['scatter', 'correlation', 'relationship']):
            return 'scatter'
        elif any(word in query_lower for word in ['horizontal', 'h-bar']):
            return 'horizontal_bar'
        
        # Use requested type if valid
        if requested_type in self.supported_types:
            return requested_type
        
        # Smart defaults based on data characteristics
        if self._is_time_series_data(graph_data):
            return 'line'
        elif 'labels' in graph_data and 'values' in graph_data:
            data_count = len(graph_data.get('values', []))
            if data_count <= 10:
                return 'pie'
            elif data_count <= 20:
                return 'bar'
            else:
                return 'horizontal_bar'
        elif 'x_values' in graph_data and 'y_values' in graph_data:
            return 'line'
        
        return 'bar'  # Default fallback
    
    def _create_graph_by_type(self, ax, graph_data: Dict, graph_type: str) -> bool:
        """Create graph based on type with complete enhanced error handling."""
        try:
            if graph_type == 'pie':
                return self._create_complete_pie_chart(ax, graph_data)
            elif graph_type == 'bar':
                return self._create_complete_bar_chart(ax, graph_data)
            elif graph_type == 'horizontal_bar':
                return self._create_complete_horizontal_bar_chart(ax, graph_data)
            elif graph_type == 'line':
                return self._create_complete_line_chart(ax, graph_data)
            elif graph_type == 'scatter':
                return self._create_complete_scatter_plot(ax, graph_data)
            else:
                logger.warning(f"Unknown graph type: {graph_type}")
                return False
        except Exception as e:
            logger.error(f"Error creating {graph_type} chart: {e}")
            return False
    
    def _create_complete_pie_chart(self, ax, graph_data: Dict) -> bool:
        """Create complete enhanced pie chart with smart data handling."""
        try:
            labels = graph_data.get('labels', graph_data.get('categories', []))
            values = graph_data.get('values', [])
            
            if not labels or not values or len(labels) != len(values):
                return False
            
            # Filter positive values and prepare data
            filtered_data = [(label, float(value)) for label, value in zip(labels, values) 
                           if isinstance(value, (int, float)) and float(value) > 0]
            
            if not filtered_data:
                ax.text(0.5, 0.5, 'No positive data to display', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                return True
            
            # Sort by value for better visualization
            filtered_data.sort(key=lambda x: x[1], reverse=True)
            
            # Limit categories if too many
            if len(filtered_data) > 8:
                top_data = filtered_data[:7]
                others_sum = sum(item[1] for item in filtered_data[7:])
                if others_sum > 0:
                    top_data.append(('Others', others_sum))
                filtered_data = top_data
            
            labels, values = zip(*filtered_data)
            
            # Create pie chart with complete enhanced styling
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                            startangle=90, colors=colors)
            
            # Complete enhance text appearance
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            for text in texts:
                text.set_fontsize(9)
            
            ax.set_aspect('equal')
            return True
            
        except Exception as e:
            logger.error(f"Complete pie chart creation failed: {e}")
            return False
    
    def _create_complete_bar_chart(self, ax, graph_data: Dict) -> bool:
        """FIXED: Create complete enhanced bar chart with smart formatting."""
        try:
            # Check for x_values/y_values first (preferred format)
            if 'x_values' in graph_data and 'y_values' in graph_data:
                x_values = graph_data['x_values']
                y_values = graph_data['y_values']
                
                if not x_values or not y_values or len(x_values) != len(y_values):
                    logger.error(f"[BAR] Invalid x/y values: x={len(x_values) if x_values else 0}, y={len(y_values) if y_values else 0}")
                    return False
                
                # Limit to reasonable number of bars
                if len(x_values) > 30:
                    x_values = x_values[:30]
                    y_values = y_values[:30]
                    logger.info("[BAR] Limited to 30 bars for readability")
                
                # Create bar chart
                bars = ax.bar(range(len(x_values)), y_values, color='steelblue', alpha=0.8, edgecolor='darkblue')
                
                # Set labels
                ax.set_xlabel(graph_data.get('x_label', 'Categories'), fontsize=12)
                ax.set_ylabel(graph_data.get('y_label', 'Values'), fontsize=12)
                ax.set_xticks(range(len(x_values)))
                ax.set_xticklabels(x_values, rotation=45, ha='right')
                
                # Add value labels on bars if not too many
                if len(x_values) <= 15:
                    max_val = max(y_values) if y_values else 1
                    for i, (bar, value) in enumerate(zip(bars, y_values)):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + max_val * 0.01,
                               f'{value:.0f}' if isinstance(value, float) else str(value),
                               ha='center', va='bottom', fontsize=8)
                
                ax.grid(True, alpha=0.3, axis='y')
                logger.info(f"[BAR] Successfully created bar chart with {len(x_values)} bars")
                return True
            
            # Fallback to categories/values format
            elif 'categories' in graph_data and 'values' in graph_data:
                categories = graph_data['categories']
                values = graph_data['values']
                
                if not categories or not values or len(categories) != len(values):
                    logger.error(f"[BAR] Invalid categories/values: cat={len(categories) if categories else 0}, val={len(values) if values else 0}")
                    return False
                
                # Limit to reasonable number
                if len(categories) > 30:
                    categories = categories[:30]
                    values = values[:30]
                    logger.info("[BAR] Limited to 30 categories for readability")
                
                bars = ax.bar(range(len(categories)), values, color='steelblue', alpha=0.8, edgecolor='darkblue')
                ax.set_xlabel(graph_data.get('x_label', 'Categories'), fontsize=12)
                ax.set_ylabel(graph_data.get('y_label', 'Values'), fontsize=12)
                ax.set_xticks(range(len(categories)))
                ax.set_xticklabels([str(cat)[:15] for cat in categories], rotation=45, ha='right')
                
                # Add value labels if not too many
                if len(categories) <= 15:
                    max_val = max(values) if values else 1
                    for i, (bar, value) in enumerate(zip(bars, values)):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + max_val * 0.01,
                               f'{value:.0f}' if isinstance(value, float) else str(value),
                               ha='center', va='bottom', fontsize=8)
                
                ax.grid(True, alpha=0.3, axis='y')
                logger.info(f"[BAR] Successfully created bar chart with {len(categories)} categories")
                return True
            
            else:
                logger.error("[BAR] Missing required data: need either (x_values, y_values) or (categories, values)")
                return False
            
        except Exception as e:
            logger.error(f"[BAR] Bar chart creation failed: {e}")
            import traceback
            logger.error(f"[BAR] Traceback: {traceback.format_exc()}")
            return False
    
    def _create_complete_horizontal_bar_chart(self, ax, graph_data: Dict) -> bool:
        """Create complete enhanced horizontal bar chart."""
        try:
            categories = graph_data.get('categories', graph_data.get('labels', []))
            values = graph_data.get('values', [])
            
            if not categories or not values or len(categories) != len(values):
                return False
            
            # Limit and sort data
            if len(categories) > 20:
                # Sort by value and take top 20
                data_pairs = list(zip(categories, values))
                data_pairs.sort(key=lambda x: float(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)
                categories, values = zip(*data_pairs[:20])
                logger.info("Limited horizontal bar chart to top 20 categories")
            
            # Create horizontal bar chart
            bars = ax.barh(range(len(categories)), values, color='lightcoral', alpha=0.8, edgecolor='darkred')
            
            # Set labels and formatting
            ax.set_xlabel(graph_data.get('x_label', 'Values'), fontsize=12)
            ax.set_ylabel(graph_data.get('y_label', 'Categories'), fontsize=12)
            ax.set_yticks(range(len(categories)))
            ax.set_yticklabels([str(cat)[:20] for cat in categories])
            
            # Add value labels
            if len(categories) <= 15:
                max_val = max(values) if values else 1
                for i, (bar, value) in enumerate(zip(bars, values)):
                    width = bar.get_width()
                    ax.text(width + max_val * 0.01, bar.get_y() + bar.get_height()/2,
                           f'{value:.1f}' if isinstance(value, float) else str(value),
                           ha='left', va='center', fontsize=8)
            
            ax.grid(True, alpha=0.3, axis='x')
            return True
            
        except Exception as e:
            logger.error(f"Complete horizontal bar chart creation failed: {e}")
            return False
    
    def _create_complete_line_chart(self, ax, graph_data: Dict) -> bool:
        """FIXED: Create line chart with proper scaling for small values."""
        try:
            # Check for x_values/y_values format
            if 'x_values' in graph_data and 'y_values' in graph_data:
                x_values = graph_data['x_values']
                y_values = graph_data['y_values']
                
                if not x_values or not y_values or len(x_values) != len(y_values):
                    logger.error(f"[LINE] Invalid x/y values: x={len(x_values) if x_values else 0}, y={len(y_values) if y_values else 0}")
                    return False
                
                # CRITICAL FIX: Check for extreme value ranges that cause scaling issues
                if len(y_values) > 1:
                    min_val = min(y_values)
                    max_val = max(y_values)
                    value_range_ratio = max_val / min_val if min_val > 0 else float('inf')
                    
                    logger.info(f"[LINE] Value range: {min_val:,.0f} to {max_val:,.0f} (ratio: {value_range_ratio:.1f})")
                    
                    # If we have extreme scaling issues (ratio > 500), use log scale or adjust
                    if value_range_ratio > 500 and min_val > 0:
                        logger.warning(f"[LINE] Extreme value range detected (ratio: {value_range_ratio:.1f})")
                        logger.info("[LINE] Applying scaling fix to prevent small values from disappearing")
                        
                        # Option 1: Use log scale if all values are positive
                        if all(val > 0 for val in y_values):
                            ax.set_yscale('log')
                            logger.info("[LINE] Applied logarithmic Y-axis scale")
                        
                        # Option 2: Set a reasonable Y-axis minimum to show small values
                        else:
                            # Set Y-axis to start from 0 but ensure small values are visible
                            y_margin = max_val * 0.05  # 5% margin
                            ax.set_ylim(bottom=0, top=max_val + y_margin)
                            logger.info(f"[LINE] Set Y-axis range: 0 to {max_val + y_margin:,.0f}")
                
                # Create the line chart
                line = ax.plot(range(len(x_values)), y_values, 
                              color='darkgreen', linewidth=2.5, marker='o', 
                              markersize=5, markerfacecolor='green', alpha=0.8)
                
                # Set labels and formatting
                ax.set_xlabel(graph_data.get('x_label', 'Time Period'), fontsize=12, fontweight='bold')
                ax.set_ylabel(graph_data.get('y_label', 'Values'), fontsize=12, fontweight='bold')
                ax.set_xticks(range(len(x_values)))
                ax.set_xticklabels(x_values, rotation=45, ha='right')
                
                # ENHANCED: Add value labels on points for better readability
                if len(x_values) <= 20:  # Only add labels if not too crowded
                    for i, (x, y) in enumerate(zip(range(len(x_values)), y_values)):
                        # Format large numbers with K, M suffixes
                        if y >= 1000000:
                            label = f'{y/1000000:.1f}M'
                        elif y >= 1000:
                            label = f'{y/1000:.0f}K'
                        else:
                            label = f'{y:.0f}'
                        # Position label above the point
                        ax.annotate(label, (x, y), textcoords="offset points", 
                                   xytext=(0,10), ha='center', fontsize=8, 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
                
                # Enhanced grid and styling
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # CRITICAL: Force the plot to show all data points properly
                ax.autoscale(tight=False)
                
                logger.info(f"[LINE] Successfully created line chart with {len(x_values)} points")
                return True
                
            else:
                logger.error("[LINE] Missing required x_values/y_values data")
                return False
                
        except Exception as e:
            logger.error(f"[LINE] Line chart creation failed: {e}")
            import traceback
            logger.error(f"[LINE] Traceback: {traceback.format_exc()}")
            return False

    def _create_dual_scale_line_chart(self, ax, graph_data: Dict) -> bool:
        """Alternative: Create line chart with dual Y-axis for extreme value ranges."""
        try:
            if 'x_values' in graph_data and 'y_values' in graph_data:
                x_values = graph_data['x_values']
                y_values = graph_data['y_values']
                
                # Detect if we need dual scale
                min_val = min(y_values)
                max_val = max(y_values)
                ratio = max_val / min_val if min_val > 0 else 1
                
                if ratio > 1000:
                    logger.info("[LINE] Using dual-scale approach for extreme value range")
                    
                    # Separate small and large values
                    threshold = max_val * 0.1  # 10% of max
                    small_indices = [i for i, val in enumerate(y_values) if val < threshold]
                    large_indices = [i for i, val in enumerate(y_values) if val >= threshold]
                    
                    if small_indices and large_indices:
                        # Create main plot for large values
                        large_x = [x_values[i] for i in large_indices]
                        large_y = [y_values[i] for i in large_indices]
                        
                        ax.plot([i for i in large_indices], large_y, 
                               color='darkgreen', linewidth=2.5, marker='o', label='High Values')
                        
                        # Create secondary axis for small values
                        ax2 = ax.twinx()
                        small_x = [x_values[i] for i in small_indices]
                        small_y = [y_values[i] for i in small_indices]
                        
                        ax2.plot([i for i in small_indices], small_y, 
                                color='orange', linewidth=2.5, marker='s', label='Low Values')
                        
                        # Set labels
                        ax.set_ylabel('High Values', color='darkgreen')
                        ax2.set_ylabel('Low Values', color='orange')
                        
                        # Add legend
                        lines1, labels1 = ax.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                        
                        return True
                
                # Fall back to regular line chart
                return self._create_complete_line_chart(ax, graph_data)
        except Exception as e:
            logger.error(f"[DUAL-LINE] Dual scale chart creation failed: {e}")
            return False
    
    def _create_complete_scatter_plot(self, ax, graph_data: Dict) -> bool:
        """Create complete enhanced scatter plot."""
        try:
            x_values = graph_data.get('x_values', [])
            y_values = graph_data.get('y_values', [])
            
            if not x_values or not y_values or len(x_values) != len(y_values):
                return False
            
            # Convert to numeric if possible
            try:
                x_numeric = [float(x) for x in x_values]
                y_numeric = [float(y) for y in y_values]
            except (ValueError, TypeError):
                return False
            
            # Handle large datasets
            if len(x_numeric) > 1000:
                step = max(1, len(x_numeric) // 500)
                x_numeric = x_numeric[::step]
                y_numeric = y_numeric[::step]
                logger.info(f"Sampled scatter plot to {len(x_numeric)} points")
            
            # Create scatter plot
            ax.scatter(x_numeric, y_numeric, alpha=0.6, s=50, color='darkblue', edgecolors='lightblue')
            
            # Set labels and formatting
            ax.set_xlabel(graph_data.get('x_label', 'X Axis'), fontsize=12)
            ax.set_ylabel(graph_data.get('y_label', 'Y Axis'), fontsize=12)
            ax.grid(True, alpha=0.3)
            
            return True
            
        except Exception as e:
            logger.error(f"Complete scatter plot creation failed: {e}")
            return False
    
    def _enhance_graph_appearance(self, fig, ax, graph_data: Dict, graph_type: str):
        """Complete enhance overall graph appearance."""
        try:
            title = graph_data.get('title', 'Data Visualization')
            description = graph_data.get('description', '')
            
            if description:
                full_title = f"{title}\n{description}"
                fig.suptitle(full_title, fontsize=14, fontweight='bold', y=0.95)
            else:
                ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Adjust layout based on graph type
            if graph_type == 'pie':
                plt.tight_layout()
            else:
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.15)
            
        except Exception as e:
            logger.warning(f"Complete graph enhancement failed: {e}")
    
    def _save_graph_safely(self, fig, graph_type: str) -> Optional[Path]:
        """Save graph with complete smart error handling."""
        try:
            timestamp = int(datetime.now().timestamp())
            filename = f"graph_{graph_type}_{timestamp}.png"
            filepath = self.graphs_dir / filename
            
            # Try primary save location
            try:
                fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                          facecolor='white', edgecolor='none')
                return filepath
            except Exception as save_error:
                # Try fallback location
                fallback_path = Path.cwd() / filename
                fig.savefig(fallback_path, dpi=300, bbox_inches='tight', 
                          facecolor='white', edgecolor='none')
                logger.info(f"Saved to fallback location: {fallback_path}")
                return fallback_path
                
        except Exception as e:
            logger.error(f"Complete graph saving failed: {e}")
            return None
    
    def _auto_open_graph(self, filepath: str) -> bool:
        """Auto-open graph with cross-platform support."""
        try:
            import subprocess
            import os
            
            if os   .name == 'nt':  # Windows
                os.startfile(filepath)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', filepath], check=True, timeout=5)
            else:  # Linux
                subprocess.run(['xdg-open', filepath], check=True, timeout=5)
            
            logger.info(f"📊 Graph opened: {filepath}")
            return True
        except Exception:
            return False

class SimpleAIModel:
    """Simple wrapper for Google AI model."""
    def __init__(self, model_name="gemini-1.5-flash"):
        import google.generativeai as genai
        self.model = genai.GenerativeModel(model_name)
    async def generate_content_async(self, prompt: str):
        try:
            response = self.model.generate_content(prompt)
            return response
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            return None

class CompleteSmartNLPProcessor:
    """COMPLETE NLP processor with enhanced threshold detection and better prompting. FIXED MULTITOOL SUPPORT."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.context = {}  # Add this line for context storage
        try:
            import google.generativeai as genai
            api_key = self.config.get('GOOGLE_API_KEY') if self.config else None
            if api_key:
                genai.configure(api_key=api_key)
            self.ai_model = SimpleAIModel()
            logger.info("✅ AI model initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI model: {e}")
            self.ai_model = None
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        self.db_schema = self._get_complete_database_schema()
        self.chart_keywords = self._get_chart_keywords()
        self.tools = self._get_tools_config()
        self.last_feedback = None
        self.last_feedback_query = None

    async def _generate_with_complete_retries(self, prompt: str, query: str, chart_analysis: Dict, max_retries: int = 3) -> List[Dict]:
        """Generate AI response with retries and better error handling"""
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"🧠 Complete AI generation attempt {attempt}")
                
                response = self.model.generate_content(prompt)
                ai_response = response.text.strip()
                logger.info(f"🧠 AI Response: {ai_response[:200]}...")
                
                # Clean up escape sequences and line breaks in JSON
                ai_response = ai_response.replace('\\\n', ' ').replace('\\n', ' ').replace('\n', ' ')
                ai_response = re.sub(r'\\+', '', ai_response)  # Remove escape sequences
                
                # Extract JSON from response
                json_match = re.search(r'```json\s*(.*?)\s*```', ai_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                    # Additional cleaning
                    json_str = re.sub(r'\s+', ' ', json_str)  # Normalize whitespace
                    try:
                        parsed_json = json.loads(json_str)
                        
                        # Handle different AI response formats
                        if isinstance(parsed_json, dict):
                            if 'tool_calls' in parsed_json:
                                # New format: {"tool_calls": [{"type": "function", "function": {"name": "...", "arguments": "..."}}]}
                                tool_calls = []
                                for tool_call in parsed_json['tool_calls']:
                                    if 'function' in tool_call:
                                        func_data = tool_call['function']
                                        tool_name = func_data.get('name', 'execute_dynamic_sql')
                                        
                                        # Normalize tool names
                                        tool_name_mapping = {
                                            'query_database': 'execute_dynamic_sql',
                                            'execute_sql': 'execute_dynamic_sql',
                                            'run_sql': 'execute_dynamic_sql',
                                            'sql_query': 'execute_dynamic_sql',
                                            'database_query': 'execute_dynamic_sql',
                                        }
                                        tool_name = tool_name_mapping.get(tool_name, tool_name)
                                        
                                        # Parse arguments (might be a string or dict)
                                        arguments = func_data.get('arguments', {})
                                        if isinstance(arguments, str):
                                            # Arguments as SQL string
                                            parameters = {'sql_query': arguments}
                                        else:
                                            parameters = arguments
                                        
                                        tool_calls.append({
                                            'tool': tool_name,
                                            'parameters': parameters,
                                            'original_query': query,
                                            'wants_graph': tool_name == 'execute_dynamic_sql_with_graph',
                                            'chart_analysis': chart_analysis or {'chart_type': 'none'}
                                        })
                                logger.info("✅ Successfully parsed tool calls")
                                return tool_calls
                            else:
                                # Single tool call as dict
                                return [parsed_json]
                        elif isinstance(parsed_json, list):
                            # Old format: [{"tool": "...", "parameters": {...}}]
                            tool_calls = []
                            for call in parsed_json:
                                if isinstance(call, dict):
                                    call['original_query'] = query
                                    call['wants_graph'] = call.get('tool') == 'execute_dynamic_sql_with_graph'
                                    call['chart_analysis'] = chart_analysis or {'chart_type': 'none'}
                                    tool_calls.append(call)
                            logger.info("✅ Successfully parsed tool calls")
                            return tool_calls
                        else:
                            logger.warning(f"Unexpected JSON format: {type(parsed_json)}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parse error: {e}")
                else:
                    logger.warning("No JSON block found in AI response")
                    
                if attempt < max_retries:
                    continue
            except Exception as e:
                logger.error(f"AI generation error: {e}")
                
        logger.warning("All AI generation attempts failed, using fallback.")
        return self._get_complete_smart_fallback_tool_call(query, [])

    def _extract_tool_calls_from_text(self, text: str, query: str, chart_analysis: dict) -> list:
        """Extract tool calls from unstructured AI response."""
        query_lower = query.lower()
        sql_patterns = [
            r'```sql\s*(SELECT.*?)```',
            r'```\s*(SELECT.*?)```',
            r'(SELECT.*?;|SELECT.*?\n\n|SELECT.*?$)',
        ]
        sql_query = None
        for pattern in sql_patterns:
            sql_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1) if pattern.startswith('```') else sql_match.group()
                sql_query = sql_query.strip().rstrip(';')
                break
        if sql_query:
            # ENFORCE: Always use graph tool if visualization is requested
            wants_graph = chart_analysis.get('wants_visualization', False) or any(word in query_lower for word in ['chart', 'graph', 'visualize', 'plot'])
            tool_name = 'execute_dynamic_sql_with_graph' if wants_graph else 'execute_dynamic_sql'
            tool_call = {
                'tool': tool_name,
                'parameters': {'sql_query': sql_query},
                'original_query': query,
                'wants_graph': wants_graph,
                'chart_analysis': chart_analysis or {'chart_type': 'none'}
            }
            if wants_graph:
                tool_call['tool'] = 'execute_dynamic_sql_with_graph'
                tool_call['wants_graph'] = True
                tool_call['parameters']['graph_type'] = chart_analysis.get('chart_type', 'bar')
            logger.info(f"🔧 Extracted SQL tool call: {tool_call['tool']} (wants_graph: {wants_graph})")
            return [tool_call]
        if 'database_status' in text.lower() or 'get_database_status' in text.lower():
            return [{
                'tool': 'get_database_status',
                'parameters': {},
                'original_query': query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'}
            }]
        if 'payment' in text.lower() and 'success' in text.lower():
            return [{
                'tool': 'get_payment_success_rate_in_last_days',
                'parameters': {'days': 30},
                'original_query': query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'}
            }]
        if 'subscription' in text.lower() and 'last' in text.lower():
            return [{
                'tool': 'get_subscriptions_in_last_days',
                'parameters': {'days': 30},
                'original_query': query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'}
            }]
        return []

    def _get_complete_database_schema(self) -> str:
        schema = """
COMPLETE ACTUAL Database Schema (Updated from Real Database):

Table: subscription_contract_v2 (2,748 total records)
AVAILABLE COLUMNS:
   - subscription_id (bigint, PRIMARY KEY, auto_increment)
   - merchant_user_id (varchar(255)) -- AVAILABLE: User identifier
   - user_email (varchar(255)) -- AVAILABLE: Customer email addresses
   - user_name (varchar(255)) -- AVAILABLE: Customer names
   - user_mobile (varchar(255)) -- AVAILABLE: Customer phone numbers
   - subcription_start_date (datetime, NOT NULL) -- TYPO: "subcription" not "subscription"
   - subcription_end_date (datetime, NOT NULL) -- TYPO: "subcription" not "subscription"
   - renewal_amount (decimal(16,2)) -- AVAILABLE: Renewal amount
   - max_amount_decimal (decimal(16,2)) -- AVAILABLE: Maximum amount
   - industry_type (varchar(255)) -- Industry classification
   - channel_id (varchar(20)) -- Channel identifier
   - service_id (varchar(100)) -- Service identifier
   - account_type (varchar(255)) -- Account type
   - auto_renewal (tinyint, DEFAULT 0) -- Auto-renewal flag
   - sub_due_date (datetime) -- AVAILABLE: Due date
   - is_new_subscription (tinyint(1), NOT NULL, DEFAULT 0) -- New subscription flag
   - plan_id (bigint) -- AVAILABLE: Plan reference
   - status (varchar(255), NOT NULL) -- Subscription status
   - created_date (datetime, DEFAULT CURRENT_TIMESTAMP)

Table: subscription_payment_details (2,840 total records, 2,748 unique subscription_ids)
AVAILABLE COLUMNS:
   - subscription_id (bigint, NOT NULL, FOREIGN KEY)
   - trans_amount_decimal (decimal(16,2)) -- AVAILABLE: Current transaction amount
   - status (varchar(255)) -- AVAILABLE: Payment status (ACTIVE, INIT, FAIL, etc.)
   - created_date (datetime, DEFAULT CURRENT_TIMESTAMP)
   - payment_time (datetime) -- AVAILABLE: Actual payment time

CRITICAL SCHEMA RULES:
- TYPOS CONFIRMED: "subcription_start_date", "subcription_end_date" (missing 's')
- EMAIL IS AVAILABLE: Use c.user_email for customer email addresses
- NAMES ARE AVAILABLE: Use c.user_name for customer names  
- AMOUNTS ARE AVAILABLE: Use c.renewal_amount, c.max_amount_decimal for subscription amounts
- PAYMENT AMOUNTS: Use p.trans_amount_decimal for payment amounts

🚨 CRITICAL FIELD SELECTION & NULL HANDLING:
- FOR SUBSCRIPTION VALUE/AMOUNTS: USE c.renewal_amount OR c.max_amount_decimal
- FOR PAYMENT REVENUE: USE p.trans_amount_decimal WHERE p.status = 'ACTIVE'
- FOR CUSTOMER VALUE: COMBINE BOTH subscription + payment amounts
- MANY user_email/user_name ARE NULL - ALWAYS use COALESCE()
- For dates with no data - use DATE RANGES (±3 days) not exact dates
"""
        # Append new critical field usage rules
        schema += """
🔥 CRITICAL FIELD USAGE RULES (NEWLY ADDED):
- FOR REVENUE QUERIES: Use p.trans_amount_decimal WHERE p.status = 'ACTIVE' 
- FOR SUBSCRIPTION VALUE: Use c.renewal_amount OR c.max_amount_decimal
- FOR USER DETAILS: Always use COALESCE(c.user_email, 'Not provided') for NULL handling
- FOR DATE QUERIES: Use DATE ranges (±3 days) if exact date returns no results
"""
        return schema

    def _get_chart_keywords(self) -> Dict[str, List[str]]:
        """Chart type keywords for better detection."""
        return {
            'pie': ['pie', 'pie chart', 'distribution', 'breakdown', 'percentage', 'proportion', 'success rate', 'payment success'],
            'line': ['line', 'trend', 'over time', 'timeline', 'time series', 'progression', 'line chart', 'line graph', 'weekly', 'monthly', 'daily'],
            'bar': ['bar', 'comparison', 'compare', 'versus', 'vs', 'bar chart', 'bar graph', 'top', 'merchants by', 'ranking'],
            'scatter': ['scatter', 'correlation', 'relationship', 'plot', 'scatter plot']
        }

    def _get_tools_config(self):
        """Get complete tools configuration."""
        return [
            genai.protos.Tool(
                function_declarations=[
                    genai.protos.FunctionDeclaration(
                        name="get_subscriptions_in_last_days",
                        description="Get subscription statistics for the last N days",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "days": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Number of days (1-365)")
                            },
                            required=["days"]
                        )
                    ),
                    genai.protos.FunctionDeclaration(
                        name="get_payment_success_rate_in_last_days",
                        description="Get payment success rate and revenue statistics for the last N days",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "days": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Number of days (1-365)")
                            },
                            required=["days"]
                        )
                    ),
                    genai.protos.FunctionDeclaration(
                        name="get_user_payment_history",
                        description="Get payment history for a specific user",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "merchant_user_id": genai.protos.Schema(type=genai.protos.Type.STRING, description="The merchant user ID"),
                                "days": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Days to look back (default: 90)")
                            },
                            required=["merchant_user_id"]
                        )
                    ),
                    genai.protos.FunctionDeclaration(
                        name="get_database_status",
                        description="Check database connection and get basic statistics",
                        parameters=genai.protos.Schema(type=genai.protos.Type.OBJECT, properties={})
                    ),
                    genai.protos.FunctionDeclaration(
                        name="execute_dynamic_sql",
                        description="Execute a custom SQL SELECT query for analytics",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "sql_query": genai.protos.Schema(type=genai.protos.Type.STRING, description="SELECT SQL query to execute")
                            },
                            required=["sql_query"]
                        )
                    ),
                    genai.protos.FunctionDeclaration(
                        name="execute_dynamic_sql_with_graph",
                        description="Execute a SQL query AND generate a graph visualization",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "sql_query": genai.protos.Schema(type=genai.protos.Type.STRING, description="SELECT SQL query to execute"),
                                "graph_type": genai.protos.Schema(type=genai.protos.Type.STRING, description="Graph type: line, bar, horizontal_bar, pie, scatter")
                            },
                            required=["sql_query"]
                        )
                    )
                ]
            )
        ]

    async def parse_query(self, user_query: str, history: List[str], client=None) -> List[Dict]:
        """FIXED MULTITOOL SUPPORT: Parse query and return list of tool calls with proper handling for multiple queries"""
        query_lower = user_query.lower().strip()
        threshold_info = self._extract_threshold_info(user_query)
        comparison_info = self._extract_comparison_info(user_query)
        is_comparison = False
        is_time_period_comparison = comparison_info.get('comparison_type') == 'time_period_comparison' and comparison_info.get('time_periods')
        if (
            (threshold_info['has_threshold'] and threshold_info['numbers'] and
            (('compare' in query_lower or 'vs' in query_lower or 'versus' in query_lower or 'and' in query_lower)))
            or is_time_period_comparison
        ):
            if is_time_period_comparison or (len(threshold_info['numbers']) >= 2):
                is_comparison = True
        # Check for business summary queries that should be split
        business_summary_patterns = [
            'business health summary', 'business summary', 'give me a summary',
            'what was', 'how many', 'what is', 'tell me'
        ]
        is_business_summary = any(pattern in query_lower for pattern in business_summary_patterns) and (query_lower.count(',') >= 2 or 'what was' in query_lower and 'how many' in query_lower)
        
        complex_patterns = [
            'list of customers', 'show me their', 'customers who have', 
            'users who have', 'find customers', 'get customers',
            'between', 'from', 'to', 'and', # Date range indicators
            'revenue', 'total', 'sum', 'amount', # Financial queries
        ]
        is_complex_analytical = any(pattern in query_lower for pattern in complex_patterns)
        is_date_range_query = any(phrase in query_lower for phrase in [
            'between', 'from', 'to', 'and', 'range', 'during'
        ]) and any(date_word in query_lower for date_word in [
            'april', 'may', 'june', 'july', 'august', 'september', 
            'october', 'november', 'december', 'january', 'february', 'march',
            '2024', '2025', 'date'
        ])
        
        # Override complex_analytical for business summary queries
        if is_business_summary:
            is_complex_analytical = False
            
        # IMPROVED: Better detection of multi-part queries with different chart types
        chart_type_indicators = ['pie chart', 'bar chart', 'line chart', 'scatter plot']
        has_multiple_chart_types = sum(1 for indicator in chart_type_indicators if indicator in query_lower) >= 2
        
        # Check for explicit multi-part queries with semicolons or "and" with different chart types
        has_explicit_separators = any(sep in user_query for sep in [';', ' and ', '\n'])
        
        # FORCE: Always split queries that contain multiple chart type indicators
        if has_multiple_chart_types:
            logger.info(f"🔧 FORCE SPLIT: Detected multiple chart types in query: {[indicator for indicator in chart_type_indicators if indicator in query_lower]}")
            is_complex_analytical = False  # Override to force splitting
        
        if not is_comparison and not is_complex_analytical and not is_date_range_query:
            # IMPROVED: Better splitting logic for multi-part queries
            query_separators = [' and ', ';', '\n']
            individual_queries = [user_query.strip()]
            
            # First pass: Split by explicit separators
            for separator in query_separators:
                new_queries = []
                for query in individual_queries:
                    parts = [q.strip() for q in query.split(separator) if q.strip()]
                    if len(parts) > 1:
                        new_queries.extend(parts)
                    else:
                        new_queries.append(query)
                individual_queries = new_queries
            
            # Second pass: If we have multiple chart type indicators, try to split more aggressively
            if has_multiple_chart_types and len(individual_queries) == 1:
                # Look for natural break points around chart type indicators
                query_text = individual_queries[0]
                chart_splits = []
                
                # Find positions of chart type indicators
                chart_positions = []
                for indicator in chart_type_indicators:
                    pos = query_text.lower().find(indicator)
                    if pos != -1:
                        chart_positions.append((pos, indicator))
                
                if len(chart_positions) >= 2:
                    # Sort by position
                    chart_positions.sort()
                    
                    # Split around chart indicators
                    last_pos = 0
                    for pos, indicator in chart_positions:
                        # Look for a good break point before this chart indicator
                        # Try to find "show me", "create", "generate" before the chart type
                        break_point = pos
                        for break_phrase in ['show me', 'create', 'generate', 'make']:
                            phrase_pos = query_text.lower().rfind(break_phrase, last_pos, pos)
                            if phrase_pos != -1:
                                break_point = phrase_pos
                                break
                        
                        if break_point > last_pos:
                            chart_splits.append(query_text[last_pos:break_point].strip())
                        last_pos = break_point
                    
                    # Add the last part
                    if last_pos < len(query_text):
                        chart_splits.append(query_text[last_pos:].strip())
                    
                    if len(chart_splits) >= 2:
                        individual_queries = [q for q in chart_splits if q.strip()]
                
                # IMPROVED: If the above didn't work, try splitting by semicolon first, then by "and"
                if len(individual_queries) == 1 and has_multiple_chart_types:
                    logger.info("🔧 FORCE SPLIT: Trying alternative splitting methods")
                    
                    # Try semicolon split first
                    if ';' in query_text:
                        parts = [part.strip() for part in query_text.split(';') if part.strip()]
                        if len(parts) >= 2:
                            individual_queries = parts
                            logger.info(f"🔧 SPLIT: Split by semicolon into {len(parts)} parts")
                    
                    # If still single query, try "and" split
                    elif len(individual_queries) == 1 and ' and ' in query_text:
                        parts = [part.strip() for part in query_text.split(' and ') if part.strip()]
                        if len(parts) >= 2:
                            individual_queries = parts
                            logger.info(f"🔧 SPLIT: Split by 'and' into {len(parts)} parts")
            
            seen = set()
            unique_queries = []
            for query in individual_queries:
                if query.lower() not in seen:
                    seen.add(query.lower())
                    unique_queries.append(query)
        elif is_business_summary:
            # Special handling for business summary queries - split by commas and clean up
            query_parts = [part.strip() for part in user_query.split(',') if part.strip()]
            # Remove the summary header and clean up each part
            cleaned_queries = []
            for part in query_parts:
                # Skip the summary header
                if any(header in part.lower() for header in ['business health summary', 'business summary', 'give me a summary']):
                    continue
                # Clean up the part
                part = part.strip()
                if part.startswith(':'):
                    part = part[1:].strip()
                # Remove leading "and" if present
                if part.lower().startswith('and '):
                    part = part[4:].strip()
                if part:
                    cleaned_queries.append(part)
            
            # If we still don't have enough parts, try splitting by "and" as well
            if len(cleaned_queries) < 3 and ' and ' in user_query:
                # Split by "and" and clean up
                and_parts = [part.strip() for part in user_query.split(' and ') if part.strip()]
                cleaned_and_queries = []
                for part in and_parts:
                    # Skip the summary header
                    if any(header in part.lower() for header in ['business health summary', 'business summary', 'give me a summary']):
                        continue
                    # Clean up the part
                    part = part.strip()
                    if part.startswith(':'):
                        part = part[1:].strip()
                    if part:
                        cleaned_and_queries.append(part)
                
                # Use the method that gives us more parts
                if len(cleaned_and_queries) > len(cleaned_queries):
                    unique_queries = cleaned_and_queries
                else:
                    unique_queries = cleaned_queries
            else:
                unique_queries = cleaned_queries
        else:
            unique_queries = [user_query.strip()]
        
        logger.info(f"🔧 MULTITOOL: Processing {len(unique_queries)} individual queries")
        all_tool_calls = []
        for i, query in enumerate(unique_queries, 1):
            logger.info(f"🔧 MULTITOOL: Processing query {i}/{len(unique_queries)}: {query[:50]}...")
            try:
                query_tool_calls = await self._process_single_query(query, history, client, force_comparison=is_comparison)
                # ENFORCE: Always use graph tool if visualization is requested
                for call in query_tool_calls:
                    chart_analysis = call.get('chart_analysis', {})
                    if (
                        (chart_analysis.get('wants_visualization') or chart_analysis.get('chart_type'))
                        and call['tool'] == 'execute_dynamic_sql'
                    ):
                        call['tool'] = 'execute_dynamic_sql_with_graph'
                        call['wants_graph'] = True
                        call['parameters']['graph_type'] = chart_analysis.get('chart_type', 'bar')
                for call in query_tool_calls:
                    call['query_index'] = i
                    call['total_queries'] = len(unique_queries)
                    call['is_multitool'] = len(unique_queries) > 1
                all_tool_calls.extend(query_tool_calls)
                logger.info(f"🔧 MULTITOOL: Query {i} generated {len(query_tool_calls)} tool calls")
            except Exception as e:
                logger.error(f"❌ MULTITOOL: Error processing query {i}: {e}")
                error_call = {
                    'tool': 'get_database_status',
                    'parameters': {},
                    'original_query': query,
                    'wants_graph': False,
                    'chart_analysis': {'chart_type': 'none'},
                    'query_index': i,
                    'total_queries': len(unique_queries),
                    'is_multitool': len(unique_queries) > 1,
                    'error': str(e)
                }
                all_tool_calls.append(error_call)
        logger.info(f"✅ MULTITOOL: Generated total of {len(all_tool_calls)} tool calls for {len(unique_queries)} queries")
        if not all_tool_calls:
            return [{
                'tool': 'get_database_status',
                'parameters': {},
                'original_query': user_query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'},
                'query_index': 1,
                'total_queries': 1,
                'is_multitool': False
            }]
        return all_tool_calls

    async def _process_single_query(self, query: str, history: List[str], client=None, auto_union=False, auto_no_graph=False, auto_chart_type=None, auto_aggregate_by=None, force_comparison=False, actionable_rules=None) -> List[Dict]:
        """Process a single query and return tool calls, with feedback-aware logic. If force_comparison is True, always generate a single UNION SQL."""
        query_lower = query.lower().strip()
        logger.info(f"[DEBUG] _process_single_query received: {query_lower}")
        
        # HANDLE CONTEXTUAL REFERENCES like "visualize that", "show that as a chart", "show in a pie chart instead", etc.
        contextual_viz_triggers = [
            'visualize that', 'show that', 'chart that', 'graph that', 'plot that',
            'visualize it', 'show it', 'chart it', 'graph it', 'plot it',
            'make a chart', 'create a graph', 'show as chart', 'show as graph',
            'show in a', 'display as', 'as a chart', 'as a graph', 'instead',
            'make graph for the same', 'graph for the same', 'chart for the same',
            'visualize the same', 'graph the same', 'chart the same'
        ]
        
        # NEW: Check for temporal modification requests that need new SQL
        temporal_modifiers = ['weekly', 'daily', 'monthly', 'hourly', 'by week', 'by day', 'by month']
        is_temporal_modification = any(modifier in query_lower for modifier in temporal_modifiers)
        
        if any(trigger in query_lower for trigger in contextual_viz_triggers):
            logger.info(f"[CONTEXT] Detected contextual visualization request: {query}")
            
            # If it's a temporal modification, don't reuse old SQL - generate new SQL with proper aggregation
            if is_temporal_modification:
                logger.info(f"[CONTEXT] Temporal modification detected: {query_lower}")
                
                # Get previous SQL context to understand what data to transform
                recent_sql_query = None
                if client and hasattr(client, 'context') and client.context.get('last_sql_query'):
                    recent_sql_query = client.context.get('last_sql_query')
                    logger.info(f"[CONTEXT] Found previous SQL for temporal modification: {recent_sql_query[:100]}...")
                
                # Store the previous context for the AI prompt
                if recent_sql_query:
                    # Determine if this was a trends/chart query
                    was_chart_query = 'SUM(' in recent_sql_query and ('time_period' in recent_sql_query or 'period' in recent_sql_query)
                    chart_instruction = " and generate a LINE CHART" if was_chart_query else ""
                    
                    temporal_context = f"TEMPORAL MODIFICATION REQUEST - Transform this previous payment trends query to {query_lower}{chart_instruction}:\nPREVIOUS SQL: {recent_sql_query}\nGENERATE: Weekly aggregated payment totals (week_period, total_value) using execute_dynamic_sql_with_graph"
                    history.append(temporal_context)
                    logger.info(f"[CONTEXT] Added temporal context to history for AI processing")
                
                # Fall through to AI processing to generate new SQL with proper temporal grouping
                pass
            else:
                recent_sql_query = None
                
                # First check if client has context
                if client and hasattr(client, 'context') and client.context.get('last_sql_query'):
                    recent_sql_query = client.context.get('last_sql_query')
                    logger.info(f"[CONTEXT] Found SQL in client context: {recent_sql_query[:100]}...")
                
                # If not in client context, look for the most recent SUCCESSFUL SQL query result in history
                if not recent_sql_query:
                    # First pass: Look specifically for comparison/category queries (UNION patterns) in stored queries
                    for line in reversed(history):
                        if 'Stored SQL query:' in line:
                            potential_sql = line.split('Stored SQL query:')[1].strip()
                            # Prioritize comparison queries with UNION or category patterns
                            if any(pattern in potential_sql.upper() for pattern in ['UNION', 'CATEGORY', 'MORE THAN']):
                                recent_sql_query = potential_sql
                                logger.info(f"[CONTEXT] Found comparison SQL in stored queries: {recent_sql_query[:50]}...")
                                break
                    
                    # Second pass: Look for UNION patterns in any line (not just stored queries)
                    if not recent_sql_query:
                        for line in reversed(history):
                            if 'UNION ALL' in line.upper() and 'SELECT' in line.upper():
                                sql_match = re.search(r'(SELECT.*?UNION ALL.*?SELECT[^;]*)', line, re.IGNORECASE | re.DOTALL)
                                if sql_match:
                                    recent_sql_query = sql_match.group(1)
                                    logger.info(f"[CONTEXT] Found UNION SQL in history: {recent_sql_query[:50]}...")
                                    break
                    
                    # Third pass: If no comparison query found, look for other non-weekly queries
                    if not recent_sql_query:
                        for line in reversed(history):
                            if 'FROM subscription_' in line and 'SELECT' in line:
                                sql_match = re.search(r'(SELECT.*?FROM subscription_[^;]*)', line, re.IGNORECASE | re.DOTALL)
                                if sql_match:
                                    potential_sql = sql_match.group(1)
                                    # Skip weekly SQL in favor of other queries
                                    if ('CONCAT(YEAR(' in potential_sql and 'WEEK(' in potential_sql):
                                        logger.info(f"[CONTEXT] Skipping weekly SQL: {potential_sql[:50]}...")
                                        continue
                                    recent_sql_query = potential_sql
                                    logger.info(f"[CONTEXT] Found non-weekly SQL in history: {recent_sql_query[:50]}...")
                                    break
                
                if recent_sql_query:
                    chart_type = auto_chart_type or 'bar'
                    if any(word in recent_sql_query.lower() for word in ['week', 'month', 'date', 'time']):
                        chart_type = 'line'
                    elif 'count' in recent_sql_query.lower() and 'group by' in recent_sql_query.lower():
                        chart_type = 'bar'
                    
                    # Override chart type based on user request
                    if 'pie' in query_lower:
                        chart_type = 'pie'
                    elif 'bar' in query_lower:
                        chart_type = 'bar'
                    elif 'line' in query_lower:
                        chart_type = 'line'
                    
                    logger.info(f"[CONTEXT] Creating {chart_type} chart from recent SQL")
                    return [{
                        'tool': 'execute_dynamic_sql_with_graph',
                        'parameters': {
                            'sql_query': recent_sql_query,
                            'graph_type': chart_type
                        },
                        'original_query': query,
                        'wants_graph': True,
                        'chart_analysis': {'chart_type': chart_type}
                    }]
        
        # HANDLE "TRY AGAIN" - IMPROVED VERSION WITH FEEDBACK EXTRACTION
        if query_lower in ['try again', 'retry', 'fix it', 'try that again']:
            # Get the most recent user query from history (exclude "try again" and feedback)
            recent_user_queries = []
            recent_feedback = None
            
            # Look through history in reverse to find the last query and any feedback
            for line in reversed(history):
                if line.startswith('User: '):
                    query_text = line[6:].strip()
                    # Skip retry commands and very short queries
                    if (not any(retry_word in query_text.lower() for retry_word in ['try again', 'retry', 'fix it']) 
                        and len(query_text) > 3):
                        recent_user_queries.append(query_text)
                        break  # Take the first (most recent) valid query
                elif ('How can this be improved?' in line or 'improvement' in line.lower()) and not recent_feedback:
                    # Extract feedback from the line after "How can this be improved?"
                    # Look for common chart type feedback patterns
                    line_lower = line.lower()
                    
                    # IMPROVED: More comprehensive chart type detection from feedback
                    if any(phrase in line_lower for phrase in ['use pie chart', 'pie chart instead', 'try pie chart', 'pie chart would be better', 'pie chart please']):
                        recent_feedback = 'pie'
                        logger.info("[TRY AGAIN] Found pie chart feedback in history")
                    elif any(phrase in line_lower for phrase in ['use bar chart', 'bar chart instead', 'try bar chart', 'bar chart would be better', 'bar chart please']):
                        recent_feedback = 'bar'
                        logger.info("[TRY AGAIN] Found bar chart feedback in history")
                    elif any(phrase in line_lower for phrase in ['use line chart', 'line chart instead', 'try line chart', 'line chart would be better', 'line chart please', 'use line graph', 'line graph instead', 'try line graph']):
                        recent_feedback = 'line'
                        logger.info("[TRY AGAIN] Found line chart feedback in history")
                    elif any(phrase in line_lower for phrase in ['use scatter', 'scatter plot', 'scatter chart']):
                        recent_feedback = 'scatter'
                        logger.info("[TRY AGAIN] Found scatter plot feedback in history")
                    # Fallback to simple pattern matching if no specific phrases found
                    elif 'pie chart' in line_lower or 'pie graph' in line_lower:
                        recent_feedback = 'pie'
                        logger.info("[TRY AGAIN] Found pie chart feedback in history (fallback)")
                    elif 'bar chart' in line_lower or 'bar graph' in line_lower:
                        recent_feedback = 'bar'
                        logger.info("[TRY AGAIN] Found bar chart feedback in history (fallback)")
                    elif 'line chart' in line_lower or 'line graph' in line_lower:
                        recent_feedback = 'line'
                        logger.info("[TRY AGAIN] Found line chart feedback in history (fallback)")
            
            if recent_user_queries:
                original_query = recent_user_queries[0]
                logger.info(f"[TRY AGAIN] Retrying with original query: {original_query}")
                if recent_feedback:
                    logger.info(f"[TRY AGAIN] Applying feedback: use {recent_feedback} chart")
                    auto_chart_type = recent_feedback
                    logger.info(f"[TRY AGAIN] auto_chart_type set to: {auto_chart_type}")
                query = original_query
                query_lower = query.lower().strip()
            else:
                logger.warning("[TRY AGAIN] No previous query found in history")
        
        chart_analysis = self._analyze_complete_chart_requirements(query, history)
        if actionable_rules is None:
            actionable_rules = []
        
        # Apply aggregation override if present
        if auto_aggregate_by:
            chart_analysis['aggregation'] = auto_aggregate_by
        
        # FIXED: Define all required variables before using them
        threshold_info = self._extract_threshold_info(query)
        date_info = self._extract_date_info(query)
        comparison_info = self._extract_comparison_info(query)
        
        # 1. Handle date range queries FIRST (between X and Y)
        if 'between' in query_lower and date_info['has_date'] and len(date_info['dates']) >= 2:
            logger.info("[DEBUG] Path: date range query detected.")
            try:
                # Parse the two dates
                date1_str = date_info['dates'][0]
                date2_str = date_info['dates'][1]
                # Convert to proper format
                if re.match(r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}', date1_str, re.IGNORECASE):
                    date1_obj = datetime.strptime(date1_str, '%d %B %Y')
                    date1_formatted = date1_obj.strftime('%Y-%m-%d')
                else:
                    date1_formatted = date1_str
                if re.match(r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}', date2_str, re.IGNORECASE):
                    date2_obj = datetime.strptime(date2_str, '%d %B %Y')
                    date2_formatted = date2_obj.strftime('%Y-%m-%d')
                else:
                    date2_formatted = date2_str
                # Detect if this is a revenue/payment query vs subscription query
                if any(word in query_lower for word in ['revenue', 'payment', 'amount', 'total', 'money', 'earnings']):
                    # Revenue query for date range
                    sql = f"""
SELECT SUM(p.trans_amount_decimal) as total_revenue, COUNT(*) as num_payments
FROM subscription_payment_details p
WHERE DATE(p.created_date) BETWEEN '{date1_formatted}' AND '{date2_formatted}' 
AND p.status = 'ACTIVE'
"""
                else:
                    # Subscription count query for date range (default)
                    sql = f"""
SELECT COUNT(*) as num_subscriptions 
FROM subscription_contract_v2 
WHERE DATE(subcription_start_date) BETWEEN '{date1_formatted}' AND '{date2_formatted}'
"""
                sql = self._fix_sql_quotes(sql)
                sql = self._validate_and_autofix_sql(sql)
                sql = self._fix_sql_date_math(sql, query)
                sql = self._fix_field_selection_issues(sql, query)
                return [{
                    'tool': 'execute_dynamic_sql',
                    'parameters': {'sql_query': sql},
                    'original_query': query,
                    'wants_graph': False,
                    'chart_analysis': {'chart_type': 'none'}
                }]
            except Exception as e:
                logger.warning(f"Error parsing date range: {e}")
                # Fall through to single date processing
        # 2. Handle specific single date queries (only if not a range)
        elif date_info['has_date'] and date_info['dates']:
            logger.info("[DEBUG] Path: specific date query detected.")
            date_str = date_info['dates'][0]
            try:
                if re.match(r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}', date_str, re.IGNORECASE):
                    date_obj = datetime.strptime(date_str, '%d %B %Y')
                    date_str = date_obj.strftime('%Y-%m-%d')
                elif re.match(r'\d{2}/\d{2}/\d{4}', date_str):
                    date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                    date_str = date_obj.strftime('%Y-%m-%d')
            except Exception:
                pass
            # ENHANCED: Detect if this is a revenue/payment query vs subscription query
            if any(word in query_lower for word in ['revenue', 'payment', 'amount', 'total', 'money', 'earnings']):
                logger.info(f"[DEBUG] Revenue query detected for date: {date_str}")
                # Revenue query for specific date
                sql = f"""
SELECT SUM(p.trans_amount_decimal) as total_revenue, COUNT(*) as num_payments
FROM subscription_payment_details p
WHERE DATE(p.created_date) BETWEEN DATE_SUB('{date_str}', INTERVAL 3 DAY) AND DATE_ADD('{date_str}', INTERVAL 3 DAY)
AND p.status = 'ACTIVE'
"""
            else:
                logger.info(f"[DEBUG] Subscription count query detected for date: {date_str}")
                # Subscription count query (default)
                sql = f"SELECT COUNT(*) as num_subscriptions FROM subscription_contract_v2 WHERE DATE(subcription_start_date) = '{date_str}'"
            sql = self._fix_sql_quotes(sql)
            sql = self._validate_and_autofix_sql(sql)
            sql = self._fix_sql_date_math(sql, query)
            sql = self._fix_field_selection_issues(sql, query)
            return [{
                'tool': 'execute_dynamic_sql',
                'parameters': {'sql_query': sql},
                'original_query': query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'}
            }]
        
        # 1b. Handle time period comparison queries (e.g., last month vs previous month)
        if comparison_info.get('comparison_type') == 'time_period_comparison' and comparison_info.get('time_periods') == ['last_month', 'prev_month']:
            logger.info("[DEBUG] Path: time period comparison detected (last month vs previous month). Generating UNION SQL.")
            # Only for revenue/payment queries
            if ('revenue' in query_lower or 'payment' in query_lower or 'total' in query_lower) and ('last month' in query_lower and ('month before' in query_lower or 'previous month' in query_lower)):
                sql = """
SELECT DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 1 MONTH), '%M %Y') AS period, SUM(p.trans_amount_decimal) AS total_revenue
FROM subscription_payment_details p
WHERE p.status = 'ACTIVE'
  AND DATE_FORMAT(p.created_date, '%Y-%m') = DATE_FORMAT(CURDATE() - INTERVAL 1 MONTH, '%Y-%m')
UNION ALL
SELECT DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 2 MONTH), '%M %Y') AS period, SUM(p.trans_amount_decimal) AS total_revenue
FROM subscription_payment_details p
WHERE p.status = 'ACTIVE'
  AND DATE_FORMAT(p.created_date, '%Y-%m') = DATE_FORMAT(CURDATE() - INTERVAL 2 MONTH, '%Y-%m')
"""
                sql = self._fix_sql_quotes(sql)
                sql = self._validate_and_autofix_sql(sql)
                sql = self._fix_sql_date_math(sql, query)
                
                return [{
                    'tool': 'execute_dynamic_sql',
                    'parameters': {'sql_query': sql},
                    'original_query': query,
                    'wants_graph': False,
                    'chart_analysis': {'chart_type': 'none'}
                }]
        
        # 2. Handle comparison queries with UNION ALL
        if force_comparison or (threshold_info['has_threshold'] and threshold_info['numbers'] and (('compare' in query_lower or 'vs' in query_lower or 'versus' in query_lower or 'and' in query_lower))):
            numbers = threshold_info['numbers']
            
            # ENHANCED: Check for complex combined conditions (subscription AND payment)
            if ('subscription' in query_lower and 'payment' in query_lower and 'and' in query_lower and ('who have' in query_lower or 'and who' in query_lower)):
                # Complex query: subscribers with X subscriptions AND Y payments
                if len(numbers) >= 2:
                    sub_threshold = numbers[0]
                    payment_threshold = numbers[1]
                    sql = f"""
SELECT COUNT(*) as num_users_meeting_both_criteria
FROM (
    SELECT c.merchant_user_id
    FROM subscription_contract_v2 c
    LEFT JOIN subscription_payment_details p ON c.subscription_id = p.subscription_id
    WHERE p.status = 'ACTIVE' OR p.status IS NULL
    GROUP BY c.merchant_user_id
    HAVING COUNT(DISTINCT c.subscription_id) > {sub_threshold} 
       AND COUNT(DISTINCT CASE WHEN p.status = 'ACTIVE' THEN p.subscription_id END) > {payment_threshold}
) combined_criteria
"""
                elif len(numbers) == 1:
                    # Same threshold for both
                    threshold = numbers[0]
                    sql = f"""
SELECT COUNT(*) as num_users_meeting_both_criteria
FROM (
    SELECT c.merchant_user_id
    FROM subscription_contract_v2 c
    LEFT JOIN subscription_payment_details p ON c.subscription_id = p.subscription_id
    WHERE p.status = 'ACTIVE' OR p.status IS NULL
    GROUP BY c.merchant_user_id
    HAVING COUNT(DISTINCT c.subscription_id) > {threshold} 
       AND COUNT(DISTINCT CASE WHEN p.status = 'ACTIVE' THEN p.subscription_id END) > {threshold}
) combined_criteria
"""
                
                sql = self._fix_sql_quotes(sql)
                sql = self._validate_and_autofix_sql(sql)
                sql = self._fix_sql_date_math(sql, query)
                
                return [{
                    'tool': 'execute_dynamic_sql',
                    'parameters': {'sql_query': sql},
                    'original_query': query,
                    'wants_graph': False,
                    'chart_analysis': {'chart_type': 'none'}
                }]
            
            # Check for mixed metrics (subscriptions and payments)
            elif ('subscription' in query_lower and 'payment' in query_lower and 
                  len(numbers) >= 1 and 'and' in query_lower and 
                  not ('who have' in query_lower or 'and who' in query_lower) and
                  ('number of' in query_lower or 'count' in query_lower or 'merchant' in query_lower or 'merhcnat' in query_lower)):
                # This is asking for separate counts: subscriptions vs payments
                threshold = numbers[0] if len(numbers) == 1 else numbers[0]
                payment_threshold = numbers[1] if len(numbers) >= 2 else threshold
                
                sql = f"""
SELECT 'More than {threshold} Subscriptions' as category, COUNT(*) as value 
FROM (SELECT merchant_user_id FROM subscription_contract_v2 GROUP BY merchant_user_id HAVING COUNT(*) > {threshold}) t1
UNION ALL
SELECT 'More than {payment_threshold} Payments' as category, COUNT(*) as value  
FROM (SELECT c.merchant_user_id FROM subscription_contract_v2 c 
      JOIN subscription_payment_details p ON c.subscription_id = p.subscription_id 
      WHERE p.status = 'ACTIVE'
      GROUP BY c.merchant_user_id HAVING COUNT(p.subscription_id) > {payment_threshold}) t2
"""
                sql = self._fix_sql_quotes(sql)
                sql = self._validate_and_autofix_sql(sql)
                sql = self._fix_sql_date_math(sql, query)
                
                return [{
                    'tool': 'execute_dynamic_sql',
                    'parameters': {'sql_query': sql},
                    'original_query': query,
                    'wants_graph': False,
                    'chart_analysis': {'chart_type': 'none'}
                }]
            # Standard comparison queries (different thresholds for same metric)
            elif len(numbers) >= 2:
                sql = f"""
SELECT 'More than {numbers[0]} Subscriptions' as category, COUNT(*) as value 
FROM (SELECT merchant_user_id FROM subscription_contract_v2 GROUP BY merchant_user_id HAVING COUNT(*) > {numbers[0]}) t1
UNION ALL
SELECT 'More than {numbers[1]} Subscriptions' as category, COUNT(*) as value  
FROM (SELECT merchant_user_id FROM subscription_contract_v2 GROUP BY merchant_user_id HAVING COUNT(*) > {numbers[1]}) t2
"""
                sql = self._fix_sql_quotes(sql)
                sql = self._validate_and_autofix_sql(sql)
                sql = self._fix_sql_date_math(sql, query)
                # Always use the graph tool for this pattern
                return [{
                    'tool': 'execute_dynamic_sql_with_graph',
                    'parameters': {'sql_query': sql, 'graph_type': 'bar'},
                    'original_query': query,
                    'wants_graph': True,
                    'chart_analysis': {'chart_type': 'bar'}
                }]
        
        # 2b. Single threshold
        if threshold_info['has_threshold'] and threshold_info['numbers']:
            threshold = threshold_info['numbers'][0]
            if 'subscription' in query_lower and ('more than' in query_lower or 'greater than' in query_lower):
                sql = f"""
SELECT COUNT(*) as num_subscribers 
FROM (
    SELECT merchant_user_id 
    FROM subscription_contract_v2 
    GROUP BY merchant_user_id 
    HAVING COUNT(*) > {threshold}
) as t
"""
                sql = self._fix_sql_quotes(sql)
                sql = self._validate_and_autofix_sql(sql)
                sql = self._fix_sql_date_math(sql, query)
                
                return [{
                    'tool': 'execute_dynamic_sql',
                    'parameters': {'sql_query': sql},
                    'original_query': query,
                    'wants_graph': False,
                    'chart_analysis': {'chart_type': 'none'}
                }]
        
        # 3. Handle visualization requests with smart chart selection
        wants_graph = any(word in query_lower for word in ['graph', 'visualize', 'bar chart', 'pie chart', 'line chart', 'show as chart', 'chart'])
        
        if auto_no_graph:
            wants_graph = False
            chart_analysis = self._analyze_complete_chart_requirements(query, history)
        
        if auto_chart_type:
            chart_analysis['chart_type'] = auto_chart_type
        
        if wants_graph:
            # FIXED: Better chart type detection and SQL generation for trends over time
            if any(word in query_lower for word in ['trend', 'over time', 'timeline']) or auto_chart_type == 'line':
                sql = """
SELECT DATE_FORMAT(p.created_date, '%M %Y') AS period, SUM(p.trans_amount_decimal) AS total_revenue
FROM subscription_payment_details p
WHERE p.status = 'ACTIVE'
  AND p.created_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
GROUP BY DATE_FORMAT(p.created_date, '%Y-%m')
ORDER BY DATE_FORMAT(p.created_date, '%Y-%m')
"""
                sql = self._fix_sql_quotes(sql)
                sql = self._validate_and_autofix_sql(sql)
                sql = self._fix_sql_date_math(sql, query)
                
                return [{
                    'tool': 'execute_dynamic_sql_with_graph',
                    'parameters': {'sql_query': sql, 'graph_type': 'line'},
                    'original_query': query,
                    'wants_graph': True,
                    'chart_analysis': chart_analysis
                }]
            
            if ('bar chart' in query_lower or 'visualize' in query_lower) and 'payment' in query_lower:
                if not any(word in query_lower for word in ['over time', 'by date', 'trend', 'daily', 'per day', 'each day', 'timeline', 'monthly', 'week']):
                    sql = """
SELECT c.merchant_user_id, COUNT(*) AS total_payments
FROM subscription_payment_details p
JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id
GROUP BY c.merchant_user_id
ORDER BY total_payments DESC
LIMIT 20
"""
                    sql = self._fix_sql_quotes(sql)
                    sql = self._validate_and_autofix_sql(sql)
                    sql = self._fix_sql_date_math(sql, query)
                    
                    return [{
                        'tool': 'execute_dynamic_sql_with_graph',
                        'parameters': {'sql_query': sql, 'graph_type': 'bar'},
                        'original_query': query,
                        'wants_graph': True,
                        'chart_analysis': chart_analysis
                    }]
            
            if 'pie chart' in query_lower and ('success' in query_lower or 'failure' in query_lower or 'rate' in query_lower):
                sql = """
SELECT 
    CASE WHEN status = 'ACTIVE' THEN 'Successful' ELSE 'Failed' END as category,
    COUNT(*) as value
FROM subscription_payment_details
GROUP BY CASE WHEN status = 'ACTIVE' THEN 'Successful' ELSE 'Failed' END
"""
                sql = self._fix_sql_quotes(sql)
                sql = self._validate_and_autofix_sql(sql)
                sql = self._fix_sql_date_math(sql, query)
                
                return [{
                    'tool': 'execute_dynamic_sql_with_graph',
                    'parameters': {'sql_query': sql, 'graph_type': 'pie'},
                    'original_query': query,
                    'wants_graph': True,
                    'chart_analysis': chart_analysis
                }]
        
        # 4. Fall back to AI processing for complex queries
        try:
            history_context = self._build_complete_history_context(history)
            improvement_context = await self._get_complete_improvement_context(query, history, client)
            similar_context = await self._get_similar_queries_context(query, client)
            
            if auto_chart_type:
                chart_analysis['chart_type'] = auto_chart_type
            
            prompt = self._create_enhanced_threshold_prompt(
                query, history_context, improvement_context, similar_context, 
                chart_analysis, threshold_info, date_info, comparison_info, actionable_rules
            )
            
            tool_calls = await self._generate_with_complete_retries(prompt, query, chart_analysis)
            
            if auto_no_graph:
                for call in tool_calls:
                    if call['tool'] == 'execute_dynamic_sql_with_graph':
                        call['tool'] = 'execute_dynamic_sql'
                        call['wants_graph'] = False
            
            if auto_chart_type:
                for call in tool_calls:
                    if call['tool'] == 'execute_dynamic_sql_with_graph':
                        prev_type = call['parameters'].get('graph_type', 'not_set')
                        call['parameters']['graph_type'] = auto_chart_type
                        logger.info(f"[AUTO-CHART-TYPE] Overriding graph_type from '{prev_type}' to '{auto_chart_type}' due to auto_chart_type setting")
            
            # --- ENFORCE CHART TYPE OVERRIDE BASED ON USER QUERY OR FEEDBACK ---
            # Detect explicit chart type requests in the query
            chart_type_override = None
            if any(word in query_lower for word in ['line graph', 'line chart']):
                chart_type_override = 'line'
            elif any(word in query_lower for word in ['bar graph', 'bar chart']):
                chart_type_override = 'bar'
            elif any(word in query_lower for word in ['pie chart', 'pie graph']):
                chart_type_override = 'pie'
            elif any(word in query_lower for word in ['scatter plot', 'scatter chart']):
                chart_type_override = 'scatter'
            
            # Also check for feedback-based override
            if not chart_type_override and auto_chart_type:
                chart_type_override = auto_chart_type
            
            # IMPROVED: Force graph tool when visualization is requested
            if chart_analysis.get('wants_visualization', False):
                logger.info(f"[ENFORCE] Visualization requested, forcing graph tool usage")
                logger.info(f"[ENFORCE] Chart analysis: {chart_analysis}")
                logger.info(f"[ENFORCE] Tool calls before conversion: {[tc['tool'] for tc in tool_calls]}")
                
                # Convert any execute_dynamic_sql to execute_dynamic_sql_with_graph
                for call in tool_calls:
                    if call['tool'] == 'execute_dynamic_sql':
                        call['tool'] = 'execute_dynamic_sql_with_graph'
                        call['wants_graph'] = True
                        logger.info(f"[ENFORCE] Converted execute_dynamic_sql to execute_dynamic_sql_with_graph")
                
                # If no graph tool calls exist, create one from the first SQL call
                if not any(call['tool'] == 'execute_dynamic_sql_with_graph' for call in tool_calls):
                    logger.info(f"[ENFORCE] No graph tool calls found, creating one from first SQL call")
                    if tool_calls and tool_calls[0]['tool'] == 'execute_dynamic_sql':
                        # Convert the first call to graph tool
                        tool_calls[0]['tool'] = 'execute_dynamic_sql_with_graph'
                        tool_calls[0]['wants_graph'] = True
                        logger.info(f"[ENFORCE] Created graph tool call from first SQL call")
                
                logger.info(f"[ENFORCE] Tool calls after conversion: {[tc['tool'] for tc in tool_calls]}")
                
                # Also set the chart type if detected
                detected_chart_type = chart_analysis.get('chart_type')
                if detected_chart_type and detected_chart_type != 'none':
                    for call in tool_calls:
                        if call['tool'] == 'execute_dynamic_sql_with_graph':
                            call['parameters']['graph_type'] = detected_chart_type
                            logger.info(f"[ENFORCE] Set graph_type to {detected_chart_type} based on chart analysis")
                else:
                    # Set default chart type if none detected
                    for call in tool_calls:
                        if call['tool'] == 'execute_dynamic_sql_with_graph':
                            call['parameters']['graph_type'] = 'bar'  # Default to bar chart
                            logger.info(f"[ENFORCE] Set default graph_type to 'bar'")
            
            if chart_type_override:
                for call in tool_calls:
                    if call['tool'] == 'execute_dynamic_sql_with_graph':
                        prev_type = call['parameters'].get('graph_type', None)
                        call['parameters']['graph_type'] = chart_type_override
                        logger.info(f"[ENFORCE] Overriding graph_type from {prev_type} to {chart_type_override} due to explicit user request or feedback.")
            
            for call in tool_calls:
                if 'sql_query' in call['parameters']:
                    call['parameters']['sql_query'] = self._fix_sql_quotes(call['parameters']['sql_query'])
                    call['parameters']['sql_query'] = self._validate_and_autofix_sql(call['parameters']['sql_query'])
                    call['parameters']['sql_query'] = self._fix_sql_date_math(call['parameters']['sql_query'], query)
                    # ENFORCE: Add LIMIT clause for "top N" requests
                    call['parameters']['sql_query'] = self._enforce_top_n_limit(call['parameters']['sql_query'], query)
            
            enhanced_calls = self._enhance_and_validate_complete_tool_calls(
                tool_calls, query, chart_analysis, threshold_info
            )
            
            logger.info(f"🧠 AI selected tool(s): {[tc['tool'] for tc in enhanced_calls]}")
            return enhanced_calls
            
        except Exception as e:
            logger.error(f"Error in AI query processing: {e}", exc_info=True)
            return self._get_complete_smart_fallback_tool_call(query, history)

    def handle_specific_date_queries(self, query: str, history: List[str]) -> List[Dict]:
        """Handle specific date queries with improved date parsing."""
        date_info = self._extract_date_info(query)
        
        if date_info['has_date'] and date_info['dates']:
            date_str = date_info['dates'][0]
            
            # Convert different date formats to YYYY-MM-DD
            try:
                if re.match(r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}', date_str, re.IGNORECASE):
                    # Parse "24 april 2025" format
                    from datetime import datetime
                    date_obj = datetime.strptime(date_str, '%d %B %Y')
                    date_str = date_obj.strftime('%Y-%m-%d')
                elif re.match(r'\d{2}/\d{2}/\d{4}', date_str):
                    # Parse "MM/DD/YYYY" format  
                    from datetime import datetime
                    date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                    date_str = date_obj.strftime('%Y-%m-%d')
                # YYYY-MM-DD format is already correct
            except Exception as e:
                logger.warning(f"Could not parse date '{date_str}': {e}")
                return []
                
            # IMPROVED: Generate the correct SQL query with better detection
            query_lower = query.lower()
            if any(word in query_lower for word in ['subscription', 'subscriptions']):
                sql = f"SELECT COUNT(*) as num_subscriptions FROM subscription_contract_v2 WHERE DATE(subcription_start_date) = '{date_str}'"
            elif any(word in query_lower for word in ['payment', 'payments', 'transaction', 'transactions']):
                sql = f"SELECT COUNT(*) as num_transactions FROM subscription_payment_details WHERE DATE(created_date) = '{date_str}'"
            else:
                # Default to subscriptions
                sql = f"SELECT COUNT(*) as num_subscriptions FROM subscription_contract_v2 WHERE DATE(subcription_start_date) = '{date_str}'"
            
            return [{
                'tool': 'execute_dynamic_sql',
                'parameters': {'sql_query': sql},
                'original_query': query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'}
            }]
        
        return []

    def _extract_threshold_info(self, query: str) -> Dict:
        """Extract threshold information from query with enhanced accuracy."""
        threshold_info = {
            'has_threshold': False,
            'numbers': [],
            'operators': [],
            'context': 'unknown'
        }
        
        query_lower = query.lower()
        
        # Extract numbers
        numbers = re.findall(r'\d+', query)
        threshold_info['numbers'] = [int(n) for n in numbers]
        
        # Detect threshold operators
        threshold_phrases = [
            'more than', 'less than', 'at least', 'or more', 'at most', 'or fewer', 'exactly', 'equal to'
        ]
        if any(phrase in query_lower for phrase in threshold_phrases):
            if 'more than' in query_lower:
                threshold_info['operators'].append('>')
            if 'less than' in query_lower:
                threshold_info['operators'].append('<')
            if 'at least' in query_lower or 'or more' in query_lower:
                threshold_info['operators'].append('>=')
            if 'at most' in query_lower or 'or fewer' in query_lower:
                threshold_info['operators'].append('<=')
            if 'exactly' in query_lower or 'equal to' in query_lower:
                threshold_info['operators'].append('=')
            threshold_info['has_threshold'] = True
        
        # Detect context
        if 'subscription' in query_lower:
            threshold_info['context'] = 'subscriptions'
        elif 'transaction' in query_lower or 'payment' in query_lower:
            threshold_info['context'] = 'transactions'
        elif 'merchant' in query_lower or 'user' in query_lower:
            threshold_info['context'] = 'users'
        
        return threshold_info

    def _extract_date_info(self, query: str) -> Dict:
        """Extract date information from query with improved parsing."""
        date_info = {
            'has_date': False,
            'dates': [],
            'date_context': 'unknown'
        }
        
        # Extract dates in various formats
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY  
            r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}',  # 24 april 2025
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                date_info['dates'].extend(matches)
                date_info['has_date'] = True
        
        # Detect date context
        query_lower = query.lower()
        if 'new' in query_lower and ('subscriber' in query_lower or 'subscription' in query_lower):
            date_info['date_context'] = 'new_subscriptions'
        elif 'payment' in query_lower or 'transaction' in query_lower:
            date_info['date_context'] = 'payments'
        elif 'last date' in query_lower or 'available' in query_lower:
            date_info['date_context'] = 'data_availability'
        
        return date_info

    def _extract_comparison_info(self, query: str) -> Dict:
        """Extract comparison information from query, including time period comparisons."""
        comparison_info = {
            'is_comparison': False,
            'elements': [],
            'comparison_type': 'unknown',
            'time_periods': []
        }
        
        query_lower = query.lower()
        
        # Detect comparison keywords
        comparison_keywords = ['compare', 'vs', 'versus', 'and', 'both', 'between', 'how does', 'difference']
        if any(keyword in query_lower for keyword in comparison_keywords):
            comparison_info['is_comparison'] = True
        
        # Detect time period comparison (e.g., last month vs the month before)
        time_periods = []
        # Loosened detection: allow for any order, extra words, etc.
        if ('last month' in query_lower and ('month before' in query_lower or 'previous month' in query_lower)) or (('month before' in query_lower or 'previous month' in query_lower) and 'last month' in query_lower):
            time_periods = ['last_month', 'prev_month']
        elif 'this month' in query_lower and 'last month' in query_lower:
            time_periods = ['this_month', 'last_month']
        
        if time_periods:
            comparison_info['is_comparison'] = True
            comparison_info['comparison_type'] = 'time_period_comparison'
            comparison_info['time_periods'] = time_periods
        
        # Extract comparison elements for threshold comparisons
        # Look for patterns like "more than 1" and "more than 2"
        threshold_matches = re.findall(r'more than (\d+)', query_lower)
        if len(threshold_matches) >= 2:
            comparison_info['is_comparison'] = True
            comparison_info['elements'] = threshold_matches
            comparison_info['comparison_type'] = 'threshold_comparison'
        
        # Look for active vs inactive patterns
        if 'active' in query_lower and ('inactive' in query_lower or 'other' in query_lower):
            comparison_info['is_comparison'] = True
            comparison_info['comparison_type'] = 'status_comparison'
        
        return comparison_info

    def _extract_actionable_rules_from_suggestions(self, improvements):
        """Extract actionable rules (aggregation, chart type, etc.) from improvement suggestions."""
        import re
        rules = []
        
        for imp in improvements:
            suggestion = imp.get('user_suggestion', '').lower()
            
            # Aggregation rules
            if re.search(r'over time.*week|trend.*week|aggregate.*week|weekly', suggestion):
                rules.append({
                    'trigger': 'over time',
                    'action': 'aggregate_by',
                    'value': 'week',
                    'instruction': "ALWAYS aggregate by week for 'over time' or trend queries."
                })
            if re.search(r'over time.*month|trend.*month|aggregate.*month|monthly', suggestion):
                rules.append({
                    'trigger': 'over time',
                    'action': 'aggregate_by',
                    'value': 'month',
                    'instruction': "ALWAYS aggregate by month for 'over time' or trend queries."
                })
            
            # Chart type rules
            if 'bar chart' in suggestion:
                rules.append({'trigger': 'bar chart', 'action': 'chart_type', 'value': 'bar', 'instruction': 'ALWAYS use a bar chart for relevant queries.'})
            if 'pie chart' in suggestion:
                rules.append({'trigger': 'pie chart', 'action': 'chart_type', 'value': 'pie', 'instruction': 'ALWAYS use a pie chart for relevant queries.'})
            if 'line chart' in suggestion or 'line graph' in suggestion:
                rules.append({'trigger': 'line chart', 'action': 'chart_type', 'value': 'line', 'instruction': 'ALWAYS use a line chart for trend queries.'})
            if 'scatter' in suggestion:
                rules.append({'trigger': 'scatter', 'action': 'chart_type', 'value': 'scatter', 'instruction': 'ALWAYS use a scatter plot for correlation/relationship queries.'})
            
            # No graph rules
            if 'do not generate a graph' in suggestion or 'no graph' in suggestion or 'no chart' in suggestion:
                rules.append({'trigger': 'no graph', 'action': 'no_graph', 'value': True, 'instruction': 'DO NOT generate a graph for this type of query.'})
        
        return rules

    def _format_chart_requirements(self, chart_analysis: Dict) -> str:
        """Format chart requirements for the AI prompt."""
        if not chart_analysis:
            return "No chart requested."
        
        chart_type = chart_analysis.get('chart_type', 'none')
        
        if chart_type == 'none':
            return "No chart requested."
        elif chart_type == 'pie':
            return """PIE CHART REQUIRED:
- Use execute_dynamic_sql_with_graph tool
- SQL must return exactly 2 columns: category (text) and value (number)
- Example: SELECT 'Successful' as category, COUNT(*) as value FROM ... UNION ALL SELECT 'Failed' as category, COUNT(*) as value FROM ..."""
        elif chart_type == 'bar':
            return """BAR CHART REQUIRED:
- Use execute_dynamic_sql_with_graph tool  
- SQL should return category and value columns
- Limit to reasonable number of bars (≤30)"""
        elif chart_type == 'line':
            return """LINE CHART REQUIRED:
- Use execute_dynamic_sql_with_graph tool
- SQL should return time/period and value columns
- Order by time ascending"""
        else:
            return f"CHART REQUIRED: {chart_type} - Use execute_dynamic_sql_with_graph tool"

    def _enhance_and_validate_complete_tool_calls(self, tool_calls: List[Dict], 
                                                 query: str, chart_analysis: Dict, 
                                                 threshold_info: Dict) -> List[Dict]:
        """Enhanced validation with field usage checking"""
        enhanced_calls = []
        for call in tool_calls:
            if 'sql_query' in call.get('parameters', {}):
                sql = call['parameters']['sql_query']
                # Apply all SQL fixes
                sql = self._fix_sql_quotes(sql)
                sql = self._validate_and_autofix_sql(sql)
                sql = self._fix_sql_date_math(sql, query)
                sql = self._fix_field_selection_issues(sql, query)
                sql = self._validate_field_usage(sql, query)  # Add this critical line
                sql = self._fix_column_name_typos(sql)  # Add new method
                # CRITICAL: Add the complete SQL schema fixing
                sql = self._fix_complete_sql_schema_issues(sql, chart_analysis, query, threshold_info)
                call['parameters']['sql_query'] = sql
            enhanced_calls.append(call)
        return enhanced_calls

    def _get_complete_smart_fallback_tool_call(self, query: str, history: List[str]) -> List[Dict]:
        """Enhanced fallback with all required keys."""
        query_lower = query.lower()
        
        # Check for detail-oriented queries that failed
        if any(phrase in query_lower for phrase in ['show me the ones', 'list the users', 'show users', 'user details']):
            # Try a simple user listing query for common cases
            if 'more than' in query_lower and any(word in query_lower for word in ['subscription', 'subscriptions']):
                # Extract threshold
                threshold_match = re.search(r'more than (\d+)', query_lower)
                threshold = int(threshold_match.group(1)) if threshold_match else 1
                
                sql = f"""
SELECT DISTINCT c.merchant_user_id, 
       c.user_email, 
       c.user_name,
       sub_count.total_subscriptions
FROM subscription_contract_v2 c
JOIN (
    SELECT merchant_user_id, COUNT(*) as total_subscriptions
    FROM subscription_contract_v2 
    GROUP BY merchant_user_id 
    HAVING COUNT(*) > {threshold}
) sub_count ON c.merchant_user_id = sub_count.merchant_user_id
ORDER BY sub_count.total_subscriptions DESC
LIMIT 20
"""
                
                return [{
                    'tool': 'execute_dynamic_sql',
                    'parameters': {'sql_query': sql},
                    'original_query': query,
                    'wants_graph': False,
                    'chart_analysis': {'chart_type': 'none'}
                }]
        
        # Original fallback logic for other cases with proper structure
        if any(word in query_lower for word in ['payment', 'transaction', 'revenue']):
            return [{
                'tool': 'get_payment_success_rate_in_last_days', 
                'parameters': {'days': 30},
                'original_query': query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'}
            }]
        elif any(word in query_lower for word in ['subscription', 'subscriber', 'user']):
            return [{
                'tool': 'get_subscriptions_in_last_days', 
                'parameters': {'days': 30},
                'original_query': query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'}
            }]
        else:
            return [{
                'tool': 'get_database_status', 
                'parameters': {},
                'original_query': query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'}
            }]

    def _create_enhanced_threshold_prompt(self, user_query: str, history_context: str, 
                                        improvement_context: str, similar_context: str, 
                                        chart_analysis: Dict, threshold_info: Dict, 
                                        date_info: Dict, comparison_info: Dict, actionable_rules=None) -> str:
        """Create enhanced prompt with better user intent detection and contextual understanding."""
        from datetime import datetime
        
        current_year = datetime.now().year
        current_month = datetime.now().strftime('%B')
        
        user_query_lower = user_query.lower()
        wants_details = any(phrase in user_query_lower for phrase in [
            'show me the ones', 'list the users', 'who are the', 'get user details',
            'show user', 'list user', 'user info', 'details', 'names', 'emails',
            'show them', 'list them', 'the actual', 'specific users', 'top', 'customers'
        ])
        
        wants_count_only = any(phrase in user_query_lower for phrase in [
            'how many', 'count of', 'number of', 'total count', 'how much'
        ]) and not wants_details
        
        detail_preference = "SHOW ACTUAL USER/RECORD DETAILS" if wants_details else "COUNT OR AGGREGATE" if wants_count_only else "PREFER DETAILS UNLESS ASKING FOR COUNTS"
        
        chart_type = chart_analysis.get('chart_type', 'none')
        if chart_type == 'pie':
            chart_requirements = """PIE CHART REQUIRED:
- Use execute_dynamic_sql_with_graph tool
- SQL must return exactly 2 columns: category (text) and value (number)
- For payment success rates: SELECT 'Successful' as category, COUNT(*) as value FROM subscription_payment_details WHERE status = 'ACTIVE' UNION ALL SELECT 'Failed' as category, COUNT(*) as value FROM subscription_payment_details WHERE status != 'ACTIVE'
- For other breakdowns: Use UNION ALL to combine categories"""
        elif chart_type == 'bar':
            chart_requirements = """BAR CHART REQUIRED:
- Use execute_dynamic_sql_with_graph tool  
- SQL should return category and value columns
- For "top N" requests: ALWAYS include LIMIT N in the SQL
- For merchant rankings: ORDER BY value DESC LIMIT N
- Limit to reasonable number of bars (≤30)"""
        elif chart_type == 'line':
            chart_requirements = """LINE CHART REQUIRED:
- Use execute_dynamic_sql_with_graph tool
- SQL should return time/period and value columns
- Order by time ascending"""
        elif chart_analysis.get('wants_visualization', False):
            chart_requirements = """VISUALIZATION REQUESTED:
- Use execute_dynamic_sql_with_graph tool
- User wants a chart/graph visualization
- Choose appropriate chart type based on data structure
- CRITICAL: MUST use execute_dynamic_sql_with_graph, NOT execute_dynamic_sql
- ALWAYS include graph_type parameter in the tool call"""
        else:
            chart_requirements = "No chart requested."

        # Add improvement context prominently at the top if available
        improvement_header = ""
        force_graph_tool = ""
        
        if improvement_context and improvement_context.strip():
            improvement_header = f"""
🚨 CRITICAL USER FEEDBACK AND IMPROVEMENT REQUESTS:
{improvement_context}

⚠️ PAY ATTENTION: The user has provided specific feedback about what they want changed. 
You MUST incorporate this feedback in your response.
"""
            
            # IMPROVED: Check if user requested a chart/graph in feedback
            improvement_lower = improvement_context.lower()
            if any(phrase in improvement_lower for phrase in ['bar chart', 'bar graph', 'pie chart', 'pie graph', 'line chart', 'line graph', 'graph', 'chart', 'visualize', 'generate a bar', 'generate a pie', 'generate a line', 'draw bar', 'draw pie', 'draw line', 'show a graph', 'generate a graph', 'gnereate a grpah']):
                force_graph_tool = """
🚨 CHART REQUESTED: User has specifically requested a chart/graph visualization.
You MUST use execute_dynamic_sql_with_graph tool, NOT execute_dynamic_sql.
CRITICAL: If user asks for a graph/chart, ALWAYS use execute_dynamic_sql_with_graph tool.
"""
        
        # IMPROVED: Extract and enforce "top N" limitations
        top_n_pattern = re.search(r'top\s+(\d+)', user_query_lower)
        limit_clause = ""
        if top_n_pattern:
            limit_number = int(top_n_pattern.group(1))
            limit_clause = f"""
🚨 TOP {limit_number} LIMITATION DETECTED:
- User specifically requested "top {limit_number}" 
- You MUST add "LIMIT {limit_number}" to your SQL query
- Do NOT return more than {limit_number} rows
- Example: SELECT ... ORDER BY ... LIMIT {limit_number}
"""

        # IMPROVED: Check for query type corrections in feedback
        query_type_correction = ""
        if improvement_context and improvement_context.strip():
            improvement_lower = improvement_context.lower()
            if any(word in improvement_lower for word in ['transaction', 'transactions']) and any(word in improvement_lower for word in ['subscription', 'subscriptions']):
                query_type_correction = """
🚨 QUERY TYPE CORRECTION DETECTED:
- User wants TRANSACTIONS (payments), NOT subscriptions
- Use subscription_payment_details table, NOT subscription_contract_v2
- Query the created_date field for transaction dates
"""
            elif any(word in improvement_lower for word in ['payment', 'payments']) and any(word in improvement_lower for word in ['subscription', 'subscriptions']):
                query_type_correction = """
🚨 QUERY TYPE CORRECTION DETECTED:
- User wants PAYMENTS, NOT subscriptions  
- Use subscription_payment_details table, NOT subscription_contract_v2
- Query the created_date field for payment dates
"""

        prompt = f"""{improvement_header}{force_graph_tool}{limit_clause}{query_type_correction}You are an expert SQL analyst for subscription data. Generate VALID JSON tool calls.

🚨 CRITICAL COLUMN NAMES (EXACT SPELLING REQUIRED):
- subcription_start_date (NOT subscription_start_date) - TYPO IS CONFIRMED
- subcription_end_date (NOT subscription_end_date) - TYPO IS CONFIRMED
- Always use table aliases: subscription_contract_v2 c, subscription_payment_details p

🔥 CRITICAL RULES:
1. ONLY use LEFT JOIN when you need payment data (p.trans_amount_decimal) - do NOT join unnecessarily
2. For user details: ALWAYS use COALESCE(c.user_email, 'Email not provided'), COALESCE(c.user_name, 'Name not provided')
3. For subscription value: Use c.renewal_amount OR c.max_amount_decimal (NOT payment amounts)
4. For revenue: Use p.trans_amount_decimal WHERE p.status = 'ACTIVE' (requires JOIN)
5. NEVER use line breaks or escape sequences in JSON strings - use single line SQL
6. For "top customers": Show merchant_user_id, email, name, and calculated value
7. For ORDER BY with COALESCE: Use the full COALESCE expression, not the alias
8. For value queries: Start with less restrictive filters - if user asks for "top customers", show ALL customers ordered by value
9. Only add restrictive filters (> 0) if specifically asked for "customers with subscriptions" or "paying customers"
10. ALWAYS add LIMIT clause when user requests "top N" - e.g., "top 5" = LIMIT 5, "top 10" = LIMIT 10
11. CRITICAL: When user requests ANY chart/graph/visualization, ALWAYS use execute_dynamic_sql_with_graph tool
12. CRITICAL: If user feedback contains "graph", "chart", "visualize", "show a graph", "generate a graph" - ALWAYS use execute_dynamic_sql_with_graph tool
13. CRITICAL: For status breakdown requests (ACTIVE, CLOSED, REJECT, INIT) - use "SELECT status as category, COUNT(*) as value FROM table GROUP BY status" - DO NOT use CASE WHEN binary classification

🔧 AVAILABLE TOOLS (USE EXACT NAMES):
- execute_dynamic_sql: for data queries without charts
- execute_dynamic_sql_with_graph: for data queries with charts
- get_subscriptions_in_last_days: basic subscription stats
- get_payment_success_rate_in_last_days: payment stats
- get_database_status: connection and basic stats

📋 REQUIRED JSON FORMAT:
```json
[
  {{
    "tool": "execute_dynamic_sql",
    "parameters": {{
      "sql_query": "SELECT ... (single line SQL)"
    }}
  }}
]
```

USER INTENT: {detail_preference}
CHART: {chart_requirements}

DATABASE SCHEMA (MySQL):
subscription_contract_v2 c: merchant_user_id, user_email, user_name, subcription_start_date, subcription_end_date, renewal_amount, max_amount_decimal, auto_renewal, status
subscription_payment_details p: subscription_id, trans_amount_decimal, status, created_date

🔧 MYSQL COMPATIBILITY RULES:
- Use DATE_FORMAT(CURRENT_DATE, '%Y-%m-01') instead of DATE_TRUNC('month', CURRENT_DATE)
- Use LAST_DAY(CURRENT_DATE) for end of month
- Use DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY) for date ranges
- Use CURRENT_DATE not CURRENT_DATE()
- JOIN: Use c.subscription_id = p.subscription_id (NOT c.merchant_user_id = p.subscription_id)

EXAMPLE CORRECT QUERIES:

✅ Top customers by subscription value (less restrictive):
```json
[{{"tool": "execute_dynamic_sql", "parameters": {{"sql_query": "SELECT c.merchant_user_id, COALESCE(c.user_email, 'Email not provided') as email, COALESCE(c.user_name, 'Name not provided') as name, COALESCE(c.renewal_amount, c.max_amount_decimal, 0) as subscription_value FROM subscription_contract_v2 c ORDER BY COALESCE(c.renewal_amount, c.max_amount_decimal, 0) DESC LIMIT 10"}}}}]
```

✅ Top 5 merchants by payment revenue (with LIMIT):
```json
[{{"tool": "execute_dynamic_sql_with_graph", "parameters": {{"sql_query": "SELECT c.merchant_user_id, COUNT(*) AS total_payments FROM subscription_payment_details p JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id WHERE p.status = 'ACTIVE' GROUP BY c.merchant_user_id ORDER BY total_payments DESC LIMIT 5", "graph_type": "bar"}}}}]
```

✅ Auto-renewal subscriptions:
```json
[{{"tool": "execute_dynamic_sql", "parameters": {{"sql_query": "SELECT COALESCE(c.user_email, 'Email not provided') as email, COALESCE(c.user_name, 'Name not provided') as name, c.subcription_end_date FROM subscription_contract_v2 c WHERE c.auto_renewal = 1 AND c.subcription_end_date IS NOT NULL ORDER BY c.subcription_end_date LIMIT 50"}}}}]
```

✅ This month subscriptions (MySQL):
```json
[{{"tool": "execute_dynamic_sql", "parameters": {{"sql_query": "SELECT COUNT(*) as new_subscriptions FROM subscription_contract_v2 c WHERE c.subcription_start_date BETWEEN DATE_FORMAT(CURRENT_DATE, '%Y-%m-01') AND LAST_DAY(CURRENT_DATE)"}}}}]
```

✅ Payment trends weekly (MySQL):
```json
[{{"tool": "execute_dynamic_sql_with_graph", "parameters": {{"sql_query": "SELECT CONCAT(YEAR(p.created_date), '-W', LPAD(WEEK(p.created_date), 2, '0')) AS week_period, SUM(p.trans_amount_decimal) AS value FROM subscription_payment_details p WHERE p.status = 'ACTIVE' AND p.created_date >= DATE_SUB(CURDATE(), INTERVAL 12 WEEK) GROUP BY CONCAT(YEAR(p.created_date), '-W', LPAD(WEEK(p.created_date), 2, '0')) ORDER BY week_period", "graph_type": "line"}}}}]
```

✅ Subscription status breakdown (individual statuses):
```json
[{{"tool": "execute_dynamic_sql_with_graph", "parameters": {{"sql_query": "SELECT status as category, COUNT(*) as value FROM subscription_contract_v2 GROUP BY status ORDER BY value DESC", "graph_type": "pie"}}}}]
```

✅ Weekly revenue from May (MySQL):
```json
[{{"tool": "execute_dynamic_sql_with_graph", "parameters": {{"sql_query": "SELECT CONCAT(YEAR(p.created_date), '-W', LPAD(WEEK(p.created_date), 2, '0')) AS week_period, SUM(p.trans_amount_decimal) AS value FROM subscription_payment_details p WHERE p.status = 'ACTIVE' AND DATE_FORMAT(p.created_date, '%Y-%m') >= '2025-05' GROUP BY CONCAT(YEAR(p.created_date), '-W', LPAD(WEEK(p.created_date), 2, '0')) ORDER BY week_period", "graph_type": "line"}}}}]
```

✅ Sample customer data:
```json
[{{"tool": "execute_dynamic_sql", "parameters": {{"sql_query": "SELECT c.merchant_user_id, COALESCE(c.user_email, 'Email not provided') as email, COALESCE(c.user_name, 'Name not provided') as name, c.renewal_amount, c.max_amount_decimal FROM subscription_contract_v2 c LIMIT 10"}}}}]
```

✅ Data exploration:
```json
[{{"tool": "execute_dynamic_sql", "parameters": {{"sql_query": "SELECT COUNT(*) as total_subscriptions, MIN(subcription_start_date) as earliest_date, MAX(subcription_start_date) as latest_date FROM subscription_contract_v2"}}}}]
```

❌ WRONG: Using subscription_start_date (missing 's')
❌ WRONG: Not using COALESCE for user_email/user_name
❌ WRONG: Using INNER JOIN (loses records with no payments)
❌ WRONG: Using tool names other than the exact ones listed above
❌ WRONG: Using PostgreSQL functions like DATE_TRUNC in MySQL
❌ WRONG: Using c.merchant_user_id = p.subscription_id (wrong JOIN condition)

Query: "{user_query}"
Generate the appropriate tool call(s):"""

        # Append critical fixes
        prompt += """
🔥 CRITICAL FIXES FOR COMMON ISSUES:

1. If date query returns no results: Use DATE ranges (±3 days)
   CORRECT: DATE(created_date) BETWEEN DATE_SUB('2025-04-24', INTERVAL 3 DAY) AND DATE_ADD('2025-04-24', INTERVAL 3 DAY)
   WRONG: DATE(created_date) = '2025-04-24'

2. For revenue queries: Use p.trans_amount_decimal WHERE p.status = 'ACTIVE'
3. For subscription value: Use c.renewal_amount OR c.max_amount_decimal  
4. Always use COALESCE for user_email and user_name (many are NULL)
5. NEVER use line breaks in JSON - write SQL as single line

🕐 TEMPORAL MODIFICATION GUIDANCE:
- If user says "weekly instead" and previous query was payment trends: Generate weekly payment totals with LINE CHART
- If user says "monthly instead" and previous query was subscription trends: Generate monthly subscription totals
- For weekly payment trends: SELECT CONCAT(YEAR(p.created_date), '-W', LPAD(WEEK(p.created_date), 2, '0')) AS week_period, SUM(p.trans_amount_decimal) AS value FROM subscription_payment_details p WHERE p.status = 'ACTIVE' GROUP BY CONCAT(YEAR(p.created_date), '-W', LPAD(WEEK(p.created_date), 2, '0')) ORDER BY week_period
- For monthly payment trends: SELECT DATE_FORMAT(p.created_date, '%Y-%m') AS month_period, SUM(p.trans_amount_decimal) AS value FROM subscription_payment_details p WHERE p.status = 'ACTIVE' GROUP BY DATE_FORMAT(p.created_date, '%Y-%m') ORDER BY month_period
- ALWAYS preserve the core metric (SUM for trends) and add graph capability when transforming trends
- DO NOT add user details (email, name) for trend queries - keep aggregated totals only
"""

        return prompt

    def _auto_fix_sql_errors(self, sql: str, error: str) -> str:
        """Enhanced auto-fix for SQL errors with GROUP BY handling."""
        import re
        
        try:
            error_lower = error.lower()
            
            # Fix MySQL GROUP BY errors (ONLY_FULL_GROUP_BY mode)
            if ('group by' in error_lower and 'not in group by clause' in error_lower) or '1055' in error:
                logger.info("🔧 Fixing MySQL GROUP BY error - rewriting as subquery")
                
                # Extract the threshold from HAVING clause if present
                threshold_match = re.search(r'HAVING COUNT\(\*\) > (\d+)', sql)
                threshold = int(threshold_match.group(1)) if threshold_match else 1
                
                # Check if this is a user detail query
                if ('user_email' in sql.lower() or 'user_name' in sql.lower()) and 'merchant_user_id' in sql.lower():
                    # Rewrite as a proper subquery to avoid GROUP BY issues
                    sql = f"""
SELECT DISTINCT c.merchant_user_id, 
       c.user_email, 
       c.user_name,
       sub_count.total_subscriptions
FROM subscription_contract_v2 c
JOIN (
    SELECT merchant_user_id, COUNT(*) as total_subscriptions
    FROM subscription_contract_v2 
    GROUP BY merchant_user_id 
    HAVING COUNT(*) > {threshold}
) sub_count ON c.merchant_user_id = sub_count.merchant_user_id
ORDER BY sub_count.total_subscriptions DESC
LIMIT 50
"""
                    logger.info(f"🔧 Rewritten as subquery with threshold {threshold} to show user details")
                    return sql.strip()
                
                # For non-user queries, try to fix by adding columns to GROUP BY
                else:
                    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
                    group_by_match = re.search(r'GROUP BY\s+(.*?)(?:\s+HAVING|\s+ORDER|\s*$)', sql, re.IGNORECASE)
                    
                    if select_match and group_by_match:
                        select_columns = [col.strip() for col in select_match.group(1).split(',')]
                        # Get base column names (without functions)
                        base_columns = []
                        for col in select_columns:
                            if '(' not in col:  # Skip aggregation functions
                                base_col = col.split(' as ')[0].strip()
                                if base_col not in ['*']:
                                    base_columns.append(base_col)
                        
                        if base_columns:
                            current_group_by = group_by_match.group(1).strip()
                            # Add missing columns to GROUP BY
                            all_group_columns = [current_group_by] + [col for col in base_columns if col not in current_group_by]
                            new_group_by = ', '.join(all_group_columns)
                            sql = re.sub(r'GROUP BY\s+.*?(?=\s+HAVING|\s+ORDER|\s*$)', 
                                       f'GROUP BY {new_group_by}', sql, flags=re.IGNORECASE)
                            logger.info(f"🔧 Updated GROUP BY to include all columns: {new_group_by}")
            
            # Fix quote escaping issues
            elif 'syntax' in error_lower or 'quote' in error_lower or '42000' in error_lower:
                logger.info("🔧 Applying quote fixes for syntax error")
                sql = sql.replace("\\'", "'").replace('\\"', '"').replace("''", "'")
                sql = sql.strip()
                
                # Fix orphaned quotes
                if sql.startswith('"') and sql.count('"') % 2 == 1:
                    sql = sql[1:]
                if sql.endswith('"') and sql.count('"') % 2 == 1:
                    sql = sql[:-1]
                
                # Fix date strings
                sql = re.sub(r'"(\d{4}-\d{2}-\d{2})', r"'\1'", sql)
                sql = re.sub(r'(\d{4}-\d{2}-\d{2})"', r"'\1'", sql)
                sql = re.sub(r'"([^"]*)"', r"'\1'", sql)
            
            # Fix unknown column errors for status values
            elif 'unknown column' in error_lower and 'status' in sql.lower():
                status_values = ['ACTIVE', 'INACTIVE', 'FAILED', 'FAIL', 'INIT', 'CLOSED', 'REJECT']
                for status in status_values:
                    pattern = rf'\bstatus\s*(?:=|!=|<>)\s*{status}\b'
                    replacement = f"status = '{status}'"
                    sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
                logger.info("🔧 Fixed status value quoting")
            
            # Clean up whitespace
            sql = re.sub(r'\s+', ' ', sql).strip()
            logger.info(f"🔧 Auto-fixed SQL: {sql[:150]}...")
            return sql
            
        except Exception as e:
            logger.warning(f"Auto-fix failed: {e}")
            return sql

    def _get_threshold_guidance(self, threshold_info: Dict, comparison_info: Dict) -> str:
        """Generate specific guidance for threshold handling."""
        if not threshold_info['has_threshold']:
            return ""
        
        guidance = ["⚠️ THRESHOLD DETECTION ALERT:"]
        
        if threshold_info['numbers']:
            guidance.append(f"- Numbers detected: {threshold_info['numbers']}")
            guidance.append(f"- ⚠️ CRITICAL: Use EXACTLY these numbers in SQL: {threshold_info['numbers']}")
        
        if threshold_info['operators']:
            guidance.append(f"- Operators detected: {threshold_info['operators']}")
        
        if comparison_info['is_comparison'] and comparison_info['comparison_type'] == 'threshold_comparison':
            guidance.append("- ⚠️ COMPARISON QUERY: User wants to compare DIFFERENT thresholds")
            guidance.append("- Use UNION ALL to show both categories separately")
            guidance.append(f"- Compare thresholds: {comparison_info['elements']}")
        
        guidance.append(f"- Context: {threshold_info['context']}")
        guidance.append("- ⚠️ CRITICAL: Use the EXACT numbers specified by the user!")
        
        return "\n".join(guidance) + "\n"

    def _get_date_guidance(self, date_info: Dict) -> str:
        """Generate specific guidance for date handling."""
        if not date_info['has_date']:
            return ""
        
        guidance = ["📅 DATE QUERY DETECTED:"]
        
        if date_info['dates']:
            guidance.append(f"- Dates found: {date_info['dates']}")
        
        guidance.append(f"- Context: {date_info['date_context']}")
        guidance.append("- ⚠️ Use single quotes for dates: '2025-04-23'")
        guidance.append("- ⚠️ Use DATE() function: DATE(created_date) = '2025-04-23'")
        guidance.append("- ⚠️ Remember column typo: subcription_start_date (not subscription_start_date)")
        
        if date_info['date_context'] == 'data_availability':
            guidance.append("- ⚠️ For 'last date available', use MAX(DATE()) queries")
        
        return "\n".join(guidance) + "\n"

    def _build_complete_history_context(self, history: List[str], user_query: str = None) -> str:
        """Build complete contextual history with smart filtering. For metric/ARPU queries, filter out unrelated feedback/history."""
        if not history:
            return "No previous context."
            
        # Smart filtering for metric queries
        metric_keywords = ['arpu', 'average revenue per user', 'average revenue', 'mean revenue', 'arppu', 'arpau', 'total revenue', 'sum', 'average', 'mean']
        is_metric_query = False
        if user_query:
            user_query_lower = user_query.lower()
            is_metric_query = any(k in user_query_lower for k in metric_keywords)
        
        recent_history = history[-6:] if len(history) > 6 else history
        context_lines = []
        
        for line in recent_history:
            if is_metric_query:
                # Only include lines relevant to metrics
                if any(k in line.lower() for k in metric_keywords):
                    context_lines.append(line)
            else:
                if any(keyword in line.lower() for keyword in ['feedback', 'improve', 'try again', 'pie chart', 'bar chart', 'line chart', 'error']):
                    context_lines.append(f"IMPORTANT: {line}")
                else:
                    context_lines.append(line)
        
        if not context_lines:
            return "No previous context."
            
        return "\n".join(context_lines)

    async def _get_complete_improvement_context(self, user_query: str, history: List[str], client, return_chart_type=False, return_rules=False):
        """Get complete improvement context, actionable rules, and best chart type."""
        try:
            improvement_lines = ["COMPLETE LEARNED IMPROVEMENTS AND CONTEXT:"]
            best_suggestion = None
            best_chart_type = None
            actionable_rules = []
            
            if history:
                recent_feedback = self._extract_complete_recent_feedback(history)
                if recent_feedback:
                    improvement_lines.append(f"RECENT USER FEEDBACK: {recent_feedback}")
            
            if client:
                try:
                    suggestions_result = await client.call_tool('get_improvement_suggestions', {
                        'original_question': user_query
                    })
                    
                    if (suggestions_result.success and 
                        suggestions_result.data and 
                        suggestions_result.data.get('improvements')):
                        improvements = suggestions_result.data['improvements'][:4]  # More for rules
                        improvement_lines.append("PAST USER IMPROVEMENTS:")
                        
                        if improvements:
                            best_suggestion = improvements[0]['user_suggestion']
                            imp = improvements[0]
                            if 'chart_type' in imp and imp['chart_type']:
                                best_chart_type = imp['chart_type']
                            else:
                                suggestion_text = best_suggestion.lower()
                                if 'bar chart' in suggestion_text:
                                    best_chart_type = 'bar'
                                elif 'pie chart' in suggestion_text:
                                    best_chart_type = 'pie'
                                elif 'line chart' in suggestion_text or 'line graph' in suggestion_text:
                                    best_chart_type = 'line'
                                elif 'scatter' in suggestion_text:
                                    best_chart_type = 'scatter'
                        
                        for improvement in improvements:
                            improvement_lines.append(f"- Issue: {improvement['user_suggestion']}")
                            improvement_lines.append(f"  Context: {improvement['similar_question']}")
                            improvement_lines.append(f"  Category: {improvement['improvement_category']}")
                        
                        # Extract actionable rules from all improvements
                        actionable_rules = self._extract_actionable_rules_from_suggestions(improvements)
                        
                except Exception as e:
                    logger.debug(f"Could not get improvement suggestions: {e}")
            
            if best_suggestion:
                improvement_lines.insert(1, f"AUTO-APPLIED IMPROVEMENT: {best_suggestion}")
            
            self._last_best_chart_type = best_chart_type
            
            if return_chart_type and return_rules:
                return ("\n".join(improvement_lines) if len(improvement_lines) > 1 else "", best_chart_type, actionable_rules)
            elif return_chart_type:
                return ("\n".join(improvement_lines) if len(improvement_lines) > 1 else "", best_chart_type)
            elif return_rules:
                return ("\n".join(improvement_lines) if len(improvement_lines) > 1 else "", actionable_rules)
            else:
                return "\n".join(improvement_lines) if len(improvement_lines) > 1 else ""
                
        except Exception as e:
            logger.warning(f"Could not get complete improvement context: {e}")
            self._last_best_chart_type = None
            if return_chart_type and return_rules:
                return ("", None, [])
            elif return_chart_type:
                return ("", None)
            elif return_rules:
                return ("", [])
            else:
                return ""

    def _inject_actionable_instructions(self, actionable_rules, user_query):
        """Return a string of imperative instructions for the prompt based on actionable rules and the current query."""
        instructions = []
        q = user_query.lower()
        
        for rule in actionable_rules:
            if rule['action'] == 'aggregate_by' and rule['trigger'] in q:
                instructions.append(rule['instruction'])
            if rule['action'] == 'chart_type' and (rule['trigger'] in q or rule['value'] in q):
                instructions.append(rule['instruction'])
            if rule['action'] == 'no_graph' and (rule['trigger'] in q or 'graph' in q or 'chart' in q):
                instructions.append(rule['instruction'])
        
        return "\n".join(instructions)

    async def _get_similar_queries_context(self, user_query: str, client) -> str:
        """Get similar successful queries for better context."""
        try:
            if not client:
                return ""
            
            similar_result = await client.call_tool('get_similar_queries', {
                'original_question': user_query
            })
            
            if (similar_result.success and 
                similar_result.data and 
                similar_result.data.get('queries')):
                
                similar_queries = similar_result.data['queries'][:2]  # Top 2
                
                context_lines = ["SIMILAR SUCCESSFUL QUERIES FOR REFERENCE:"]
                for query in similar_queries:
                    context_lines.append(f"- Similar Q: {query['question']}")
                    context_lines.append(f"  Successful SQL: {query['successful_sql']}")
                    context_lines.append(f"  Category: {query['query_category']}")
                
                return "\n".join(context_lines)
            
            return ""
            
        except Exception as e:
            logger.debug(f"Could not get similar queries context: {e}")
            return ""

    def _extract_complete_recent_feedback(self, history: List[str]) -> str:
        """Extract complete recent feedback from conversation history."""
        try:
            # Look for feedback in last few turns
            for line in reversed(history[-6:]):  # Increased search range
                line_lower = line.lower()
                
                # IMPROVED: More comprehensive chart type feedback detection
                # Check for specific improvement suggestions first
                if any(phrase in line_lower for phrase in ['use pie chart', 'pie chart instead', 'try pie chart', 'pie chart would be better', 'pie chart please']):
                    return "User specifically requested PIE CHART visualization"
                elif any(phrase in line_lower for phrase in ['use bar chart', 'bar chart instead', 'try bar chart', 'bar chart would be better', 'bar chart please']):
                    return "User specifically requested BAR CHART visualization"
                elif any(phrase in line_lower for phrase in ['use line chart', 'line chart instead', 'try line chart', 'line chart would be better', 'line chart please', 'use line graph', 'line graph instead', 'try line graph']):
                    return "User specifically requested LINE CHART visualization"
                elif any(phrase in line_lower for phrase in ['use scatter', 'scatter plot', 'scatter chart']):
                    return "User specifically requested SCATTER PLOT visualization"
                
                # IMPROVED: Check for query type corrections
                elif any(word in line_lower for word in ['transaction', 'transactions']) and any(word in line_lower for word in ['subscription', 'subscriptions']):
                    return "User is correcting query type - they want TRANSACTIONS, not subscriptions"
                elif any(word in line_lower for word in ['payment', 'payments']) and any(word in line_lower for word in ['subscription', 'subscriptions']):
                    return "User is correcting query type - they want PAYMENTS, not subscriptions"
                # Fallback to simple pattern matching
                elif 'pie chart' in line_lower or 'pie graph' in line_lower:
                    return "User specifically requested PIE CHART visualization"
                elif 'bar chart' in line_lower or 'bar graph' in line_lower:
                    return "User specifically requested BAR CHART visualization"
                elif 'line chart' in line_lower or 'line graph' in line_lower:
                    return "User specifically requested LINE CHART visualization"
                elif 'scatter' in line_lower:
                    return "User specifically requested SCATTER PLOT visualization"
                elif 'improve' in line_lower and ('rate' in line_lower or 'success' in line_lower):
                    return "User wants success/failure rate analysis"
                elif 'try again' in line_lower:
                    return "User wants to retry with previous feedback incorporated"
                elif 'error' in line_lower or 'wrong' in line_lower:
                    return "Previous query had errors - user wants corrected version"
                elif 'merchant' in line_lower and 'transaction' in line_lower:
                    return "User asking about merchant transaction analysis"
                elif 'threshold' in line_lower or 'number' in line_lower:
                    return "User wants specific threshold/number analysis"
            
            return ""
        except Exception as e:
            logger.debug(f"Could not extract recent feedback: {e}")
            return ""

    def _analyze_complete_chart_requirements(self, user_query: str, history: List[str]) -> Dict:
        """Analyze complete chart/visualization requirements."""
        query_lower = user_query.lower()
        analysis = {
            'wants_visualization': False,
            'chart_type': None,
            'data_aggregation': None,
            'specific_request': None,
            'is_merchant_analysis': False,
            'needs_success_failure_breakdown': False
        }
        
        # IMPROVED: Check for feedback in parentheses (from recursive feedback loop)
        feedback_match = re.search(r'\((.*?)\)', user_query)
        if feedback_match:
            feedback_text = feedback_match.group(1).lower()
            logger.info(f"[CHART] Found feedback in parentheses: '{feedback_text}'")
            
            # Check feedback for chart requests
            if any(phrase in feedback_text for phrase in ['bar chart', 'bar graph', 'generate bar', 'draw bar', 'bar graph showing']):
                analysis['chart_type'] = 'bar'
                analysis['wants_visualization'] = True
                analysis['specific_request'] = f"User specifically requested BAR CHART in feedback: '{feedback_text}'"
                logger.info(f"[CHART] Detected BAR CHART request in feedback")
            elif any(phrase in feedback_text for phrase in ['pie chart', 'pie graph', 'generate pie', 'draw pie']):
                analysis['chart_type'] = 'pie'
                analysis['wants_visualization'] = True
                analysis['specific_request'] = f"User specifically requested PIE CHART in feedback: '{feedback_text}'"
                logger.info(f"[CHART] Detected PIE CHART request in feedback")
            elif any(phrase in feedback_text for phrase in ['line chart', 'line graph', 'generate line', 'draw line']):
                analysis['chart_type'] = 'line'
                analysis['wants_visualization'] = True
                analysis['specific_request'] = f"User specifically requested LINE CHART in feedback: '{feedback_text}'"
                logger.info(f"[CHART] Detected LINE CHART request in feedback")
            elif any(phrase in feedback_text for phrase in ['scatter', 'scatter plot', 'generate scatter']):
                analysis['chart_type'] = 'scatter'
                analysis['wants_visualization'] = True
                analysis['specific_request'] = f"User specifically requested SCATTER PLOT in feedback: '{feedback_text}'"
                logger.info(f"[CHART] Detected SCATTER PLOT request in feedback")
            elif any(phrase in feedback_text for phrase in ['chart', 'graph', 'visualize', 'draw', 'show a graph', 'generate a graph', 'gnereate a grpah']):
                analysis['wants_visualization'] = True
                analysis['specific_request'] = f"User requested visualization in feedback: '{feedback_text}'"
                logger.info(f"[CHART] Detected general visualization request in feedback")
        
        # Check for visualization keywords in main query
        viz_keywords = ['chart', 'graph', 'plot', 'visualize', 'show', 'display', 'visually']
        if any(keyword in query_lower for keyword in viz_keywords):
            analysis['wants_visualization'] = True
        
        # Detect specific chart types with improved detection (only if not already detected from feedback)
        if not analysis['chart_type']:
            for chart_type, keywords in self.chart_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    analysis['chart_type'] = chart_type
                    # Store the specific keyword that matched for better context
                    matched_keyword = next((kw for kw in keywords if kw in query_lower), None)
                    if matched_keyword:
                        analysis['specific_request'] = f"User requested {chart_type.upper()} chart via '{matched_keyword}'"
                    break
        
        # Check for merchant analysis
        if 'merchant' in query_lower and ('transaction' in query_lower or 'payment' in query_lower):
            analysis['is_merchant_analysis'] = True
        
        # Check for success/failure analysis
        if any(word in query_lower for word in ['success', 'failure', 'rate']):
            analysis['needs_success_failure_breakdown'] = True
        
        # Check for specific status breakdown requests
        if any(word in query_lower for word in ['status', 'active', 'closed', 'reject', 'init']) and 'breakdown' in query_lower:
            analysis['needs_status_breakdown'] = True
        
        # Check history for chart requests
        if not analysis['chart_type'] and history:
            for line in reversed(history[-3:]):
                if 'pie chart' in line.lower():
                    analysis['chart_type'] = 'pie'
                    analysis['specific_request'] = "User previously requested pie chart"
                    break
                elif 'line chart' in line.lower() or 'line graph' in line.lower():
                    analysis['chart_type'] = 'line'
                    analysis['specific_request'] = "User previously requested line chart"
                    break
                elif 'bar chart' in line.lower():
                    analysis['chart_type'] = 'bar'
                    analysis['specific_request'] = "User previously requested bar chart"
                    break
        
        # Determine data aggregation needs
        if analysis['chart_type'] == 'pie':
            if analysis['is_merchant_analysis']:
                analysis['data_aggregation'] = 'merchant_aggregation'
            elif analysis['needs_success_failure_breakdown']:
                analysis['data_aggregation'] = 'success_failure_breakdown'
            else:
                analysis['data_aggregation'] = 'total_summary'
        elif 'trend' in query_lower or 'over time' in query_lower or 'weekly' in query_lower or 'monthly' in query_lower or 'daily' in query_lower:
            analysis['data_aggregation'] = 'time_series'
            # Override chart type for time series data
            if analysis['chart_type'] == 'pie':
                analysis['chart_type'] = 'line'
                analysis['specific_request'] = "Time series data detected - switching from pie to line chart"
        elif 'rate' in query_lower or 'percentage' in query_lower:
            analysis['data_aggregation'] = 'rate_calculation'
        
        logger.info(f"[CHART] Final chart analysis: {analysis}")
        return analysis

    def _get_complete_chart_guidance(self, chart_analysis: Dict) -> str:
        """Get complete specific guidance based on chart analysis."""
        if not chart_analysis['wants_visualization']:
            return ""
        
        guidance = ["COMPLETE CHART REQUIREMENTS DETECTED:"]
        
        if chart_analysis['chart_type']:
            guidance.append(f"- Requested chart type: {chart_analysis['chart_type'].upper()}")
        
        if chart_analysis['data_aggregation']:
            guidance.append(f"- Data aggregation needed: {chart_analysis['data_aggregation']}")
        
        if chart_analysis['specific_request']:
            guidance.append(f"- IMPORTANT: {chart_analysis['specific_request']}")
        
        if chart_analysis['is_merchant_analysis']:
            guidance.append("- MERCHANT ANALYSIS: Use proper JOINs and categorization")
        
        if chart_analysis['needs_success_failure_breakdown']:
            guidance.append("- SUCCESS/FAILURE: Use aggregated success vs failure analysis")
        
        if chart_analysis.get('needs_status_breakdown', False):
            guidance.append("- STATUS BREAKDOWN: Show individual status values (ACTIVE, CLOSED, REJECT, INIT)")
            guidance.append("- USE: GROUP BY status to show each status separately")
            guidance.append("- DO NOT: Group into binary success/failure categories")
            guidance.append("- SQL PATTERN: SELECT status as category, COUNT(*) as value FROM table GROUP BY status")
            guidance.append("- CRITICAL: Do NOT use CASE WHEN status = 'ACTIVE' THEN 'Successful' ELSE 'Failed'")
        
        if chart_analysis['chart_type'] == 'pie':
            guidance.append("- PIE CHART REQUIRES: Aggregated summary data with categories and totals")
            guidance.append("- DO NOT use time series data for pie charts")
            guidance.append("- USE: UNION or proper GROUP BY to create category/value pairs")
            guidance.append("- CRITICAL: Follow complete schema rules for merchant_user_id access")
            guidance.append("- FOR MERCHANTS: Use subqueries to categorize merchants by activity/success")
        elif chart_analysis['chart_type'] == 'line':
            guidance.append("- LINE CHART REQUIRES: Time series data with period/date and values")
            guidance.append("- USE: DATE_FORMAT for time grouping (e.g., '%M %Y' for monthly)")
            guidance.append("- ORDER BY: Always order by time/period for proper line progression")
            guidance.append("- CRITICAL: Ensure data has time dimension for line charts")
        
        return "\n".join(guidance) + "\n"

    def _fix_complete_sql_schema_issues(self, sql_query: str, chart_analysis: Dict, 
                                      user_query: str, threshold_info: Dict = None) -> str:
        """Fix SQL with enhanced threshold handling and schema compliance."""
        try:
            # Clean SQL first
            sql_query = sql_query.strip().strip('"\'')

            # If the SQL is a simple count, do not modify it
            if re.match(r"SELECT\s+COUNT\(\*\)", sql_query, re.IGNORECASE):
                return sql_query
            
            # ENHANCED: Verify threshold accuracy
            if threshold_info and threshold_info['has_threshold'] and threshold_info['numbers']:
                sql_query = self._verify_and_fix_thresholds(sql_query, threshold_info, user_query)
            
            # Handle date queries specifically
            if 'last date' in user_query.lower() or 'available' in user_query.lower():
                if 'MAX' not in sql_query.upper():
                    # Convert to MAX date query
                    sql_query = """
SELECT MAX(DATE(created_date)) as last_payment_date FROM subscription_payment_details
UNION ALL
SELECT MAX(DATE(subcription_start_date)) as last_subscription_date FROM subscription_contract_v2
"""
            
            # Handle status breakdown queries
            if chart_analysis.get('needs_status_breakdown', False):
                logger.warning("🔧 Fixing status breakdown query to show individual statuses")
                # Check if the current SQL is using binary classification instead of individual statuses
                if 'CASE WHEN status' in sql_query and 'Successful' in sql_query and 'Failed' in sql_query:
                    logger.warning("🔧 Detected binary status classification, converting to individual status breakdown")
                
                if 'subscription' in user_query.lower() or 'subscription_contract_v2' in sql_query:
                    sql_query = """
SELECT 
    status as category,
    COUNT(*) as value
FROM subscription_contract_v2 
GROUP BY status
ORDER BY value DESC
"""
                elif 'payment' in user_query.lower() or 'subscription_payment_details' in sql_query:
                    sql_query = """
SELECT 
    status as category,
    COUNT(*) as value
FROM subscription_payment_details 
GROUP BY status
ORDER BY value DESC
"""
            
            # Original schema fixes...
            if 'GROUP BY merchant_user_id' in sql_query and 'subscription_payment_details' in sql_query:
                logger.warning("🔧 Fixing merchant_user_id GROUP BY issue with complete logic")
                
                if chart_analysis.get('chart_type') == 'pie' and 'merchant' in user_query.lower():
                    # Use threshold from query if available
                    threshold = 1
                    if threshold_info and threshold_info['numbers']:
                        threshold = threshold_info['numbers'][0]
                    
                    sql_query = f"""
SELECT 
  CASE WHEN total_transactions > {threshold} THEN 'More than {threshold} Transactions' ELSE '{threshold} or Fewer Transactions' END as category,
  COUNT(*) as value
FROM (
  SELECT c.merchant_user_id, COUNT(*) as total_transactions
  FROM subscription_payment_details p 
  JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id
  GROUP BY c.merchant_user_id
) merchant_stats
GROUP BY CASE WHEN total_transactions > {threshold} THEN 'More than {threshold} Transactions' ELSE '{threshold} or Fewer Transactions' END
"""
            
            # Apply other fixes...
            sql_query = self._apply_complete_general_sql_optimizations(sql_query)
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Complete SQL fixing failed: {e}")
            return sql_query

    def _verify_and_fix_thresholds(self, sql_query: str, threshold_info: Dict, user_query: str) -> str:
        """Verify and fix threshold numbers in SQL with enhanced accuracy."""
        try:
            # Extract threshold from user query
            user_numbers = threshold_info['numbers']
            if not user_numbers:
                return sql_query
            
            # Find threshold numbers in SQL
            sql_numbers = re.findall(r'>\s*(\d+)|<\s*(\d+)|=\s*(\d+)', sql_query)
            sql_threshold_numbers = []
            for match in sql_numbers:
                for group in match:
                    if group:
                        sql_threshold_numbers.append(int(group))
            
            # Check if SQL uses wrong threshold
            if sql_threshold_numbers and user_numbers:
                expected_threshold = user_numbers[0]  # Use first number found
                actual_threshold = sql_threshold_numbers[0]  # Use first threshold found
                
                if actual_threshold != expected_threshold:
                    logger.warning(f"🔧 Fixing threshold: SQL uses {actual_threshold}, user asked for {expected_threshold}")
                    # Replace the wrong threshold with correct one
                    sql_query = re.sub(rf'>\s*{actual_threshold}\b', f'> {expected_threshold}', sql_query)
                    sql_query = re.sub(rf'<\s*{actual_threshold}\b', f'< {expected_threshold}', sql_query)
                    sql_query = re.sub(rf'=\s*{actual_threshold}\b', f'= {expected_threshold}', sql_query)
                    
                    # Also fix in text labels
                    sql_query = re.sub(rf'More than {actual_threshold}', f'More than {expected_threshold}', sql_query)
                    sql_query = re.sub(rf'{actual_threshold} or Fewer', f'{expected_threshold} or Fewer', sql_query)
                    sql_query = re.sub(rf'{actual_threshold} or Less', f'{expected_threshold} or Less', sql_query)
            
            return sql_query
            
        except Exception as e:
            logger.warning(f"Threshold verification failed: {e}")
            return sql_query

    def _apply_complete_general_sql_optimizations(self, sql_query: str) -> str:
        """Apply complete general SQL optimizations."""
        # Clean quotes safely
        sql_query = sql_query.replace("\\'", "'")  # Remove escaped quotes first
        sql_query = sql_query.replace('\\"', '"')  # Remove escaped double quotes
        
        # Fix quotes more carefully
        sql_query = re.sub(r'"([^"\']*)"', r"'\1'", sql_query)
        
        # Fix status values carefully
        status_values = ['ACTIVE', 'INACTIVE', 'FAILED', 'FAIL', 'INIT', 'CLOSED', 'REJECT']
        for status in status_values:
            # Only fix if not already quoted
            pattern = rf'\bstatus\s*=\s*{status}\b'
            replacement = f"status = '{status}'"
            sql_query = re.sub(pattern, replacement, sql_query, flags=re.IGNORECASE)
        
        # Clean whitespace
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()
        
        return sql_query

    def _fix_sql_quotes(self, sql_query: str) -> str:
        """Replace all double-quoted string literals with single quotes for MySQL compatibility."""
        import re
        # Replace = "SOMETHING" with = 'SOMETHING'
        sql_query = re.sub(r'=\s*"([^"]*)"', r"= '\1'", sql_query)
        # Replace "SOMETHING" in WHERE/IN clauses
        sql_query = re.sub(r'IN\s*\(([^)]*)\)', lambda m: 'IN (' + ', '.join([f"'{x.strip().strip('"')}'" if x.strip().startswith('"') and x.strip().endswith('"') else x for x in m.group(1).split(',')]) + ')', sql_query)
        return sql_query

    def _validate_and_autofix_sql(self, sql_query: str) -> str:
        """Validate and auto-fix common SQL syntax issues: parentheses, IN clauses, dangling literals, and unclosed quotes."""
        import re
        fixed = False
        
        # 0. CRITICAL FIX: Fix malformed weekly GROUP BY clauses
        if 'WEEK(' in sql_query and 'GROUP BY' in sql_query:
            # Fix malformed GROUP BY where two patterns got merged incorrectly
            malformed_pattern = r'GROUP BY CONCAT\([^)]+\)[A-Z]*\([^)]+\), [A-Z]*\([^)]+\)'
            if re.search(malformed_pattern, sql_query):
                logger.info("🔧 Fixing malformed weekly GROUP BY clause")
                # Use the CONCAT expression for weekly aggregation
                concat_match = re.search(r'CONCAT\(YEAR\([^)]+\),\s*[^,]+,\s*LPAD\(WEEK\([^)]+\),\s*[^)]+\)\)', sql_query)
                if concat_match:
                    concat_expr = concat_match.group(0)
                    sql_query = re.sub(
                        malformed_pattern,
                        f"GROUP BY {concat_expr}",
                        sql_query
                    )
                else:
                    # Fallback to old pattern if CONCAT not found
                    sql_query = re.sub(
                        malformed_pattern,
                        "GROUP BY YEAR(p.created_date), WEEK(p.created_date)",
                        sql_query
                    )
                fixed = True
        
        # 1. Balance parentheses
        open_parens = sql_query.count('(')
        close_parens = sql_query.count(')')
        if open_parens > close_parens:
            sql_query += ')' * (open_parens - close_parens)
            fixed = True
        elif close_parens > open_parens:
            sql_query = '(' * (close_parens - open_parens) + sql_query
            fixed = True
        
        # 2. Ensure IN (...) clauses are closed
        in_pattern = re.compile(r'IN\s*\(([^)]*)$', re.IGNORECASE)
        match = in_pattern.search(sql_query)
        if match:
            sql_query += ')'
            fixed = True
        
        # 3. Warn if SQL ends with a dangling string literal
        if re.search(r"'\w+'\s*$", sql_query) and not sql_query.strip().lower().endswith("'as value"):
            logger.warning("SQL ends with a string literal; possible missing clause or context.")
            # Only log, do not print to user
        
        # 4. Ensure all quotes are closed
        sql_query = self._ensure_closed_quotes(sql_query)
        
        if fixed:
            logger.warning(f"SQL auto-fixed for syntax: {sql_query}")
            # Only log, do not print to user
        
        # Fix payment status values to match schema
        sql_query = re.sub(r"status\s*=\s*'success(ful)?'", "status = 'ACTIVE'", sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r"status\s*!=\s*'success(ful)?'", "status != 'ACTIVE'", sql_query, flags=re.IGNORECASE)
        
        return sql_query

    def _enforce_top_n_limit(self, sql_query: str, user_query: str) -> str:
        """Enforce LIMIT clause when 'top N' is requested."""
        import re
        
        # Check for "top N" pattern in user query
        top_n_match = re.search(r'top\s+(\d+)', user_query.lower())
        if not top_n_match:
            return sql_query
        
        limit_number = int(top_n_match.group(1))
        
        # Check if LIMIT is already present
        if re.search(r'LIMIT\s+\d+', sql_query, re.IGNORECASE):
            # Update existing LIMIT to the requested number
            sql_query = re.sub(r'LIMIT\s+\d+', f'LIMIT {limit_number}', sql_query, flags=re.IGNORECASE)
        else:
            # Add LIMIT clause at the end
            sql_query = sql_query.rstrip().rstrip(';') + f' LIMIT {limit_number}'
        
        logger.info(f"🔧 Enforced LIMIT {limit_number} for 'top {limit_number}' request")
        return sql_query

    def _fix_sql_date_math(self, sql_query: str, user_query: str = None) -> str:
        """Convert SQLite-style date math to MySQL-compatible syntax. Handles both single and double quotes and all common intervals."""
        import re
        from datetime import datetime
        
        # Replace DATE('now', '-N day') or DATE("now", '-N day') with DATE_SUB(CURDATE(), INTERVAL N DAY)
        sql_query = re.sub(r"DATE\(['\"]now['\"],\s*'-?(\d+) day'\)", r"DATE_SUB(CURDATE(), INTERVAL \1 DAY)", sql_query)
        # Replace DATE('now', '-N month') or DATE("now", '-N month') with DATE_SUB(CURDATE(), INTERVAL N MONTH)
        sql_query = re.sub(r"DATE\(['\"]now['\"],\s*'-?(\d+) month'\)", r"DATE_SUB(CURDATE(), INTERVAL \1 MONTH)", sql_query)
        # Replace DATE('now', '-N year') or DATE("now", '-N year') with DATE_SUB(CURDATE(), INTERVAL N YEAR)
        sql_query = re.sub(r"DATE\(['\"]now['\"],\s*'-?(\d+) year'\)", r"DATE_SUB(CURDATE(), INTERVAL \1 YEAR)", sql_query)
        # Replace DATE('now') or DATE("now") with CURDATE()
        sql_query = re.sub(r"DATE\(['\"]now['\"]\)", "CURDATE()", sql_query)

        # FIXED: Handle month-only queries to default to current year
        if user_query:
            # Check for month name without year
            month_only_match = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', user_query.lower())
            year_mentioned = re.search(r'20\d{2}', user_query)
            
            if month_only_match and not year_mentioned:
                # User mentioned month but no year - default to current year
                month_str = month_only_match.group(1)
                current_year = datetime.now().year
                
                month_map = {
                    'january': '01', 'february': '02', 'march': '03', 'april': '04', 'may': '05', 'june': '06',
                    'july': '07', 'august': '08', 'september': '09', 'october': '10', 'november': '11', 'december': '12'
                }
                month_num = month_map[month_str]
                date_filter = f"WHERE DATE_FORMAT(p.created_date, '%Y-%m') = '{current_year}-{month_num}'"
                
                # CRITICAL FIX: Only add if not already present AND no existing WHERE with specific date
                # ALSO: Don't modify weekly/temporal aggregation SQL that already has proper date filtering
                # ALSO: Don't modify revenue queries that already have DATE_SUB filters
                if (date_filter not in sql_query and 
                    'WHERE DATE(' not in sql_query and 
                    f"= '{current_year}-{month_num}" not in sql_query and
                    'WEEK(' not in sql_query and  # Don't modify weekly SQL
                    'DATE_FORMAT(p.created_date, \'%Y-%m\')' not in sql_query and  # Don't modify if already has month filter
                    'DATE_SUB(' not in sql_query and  # Don't modify revenue queries with DATE_SUB
                    'DATE_ADD(' not in sql_query):  # Don't modify revenue queries with DATE_ADD
                    
                    # FIXED: Properly insert date filter without creating duplicate WHERE clauses
                    if 'WHERE' in sql_query:
                        # Add to existing WHERE clause with AND
                        sql_query = re.sub(r'WHERE\s+', f'WHERE {date_filter} AND ', sql_query, count=1)
                    else:
                        # Add new WHERE clause
                        if 'GROUP BY' in sql_query:
                            sql_query = sql_query.replace('GROUP BY', f'WHERE {date_filter} GROUP BY')
                        else:
                            sql_query += f' WHERE {date_filter}'
            
            elif year_mentioned:
                # User mentioned both month and year - use the specified year
                match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+20\d{2}', user_query.lower())
                if match:
                    month_str = match.group(1)
                    year_str = re.search(r'20\d{2}', user_query).group(0)
                    month_map = {
                        'january': '01', 'february': '02', 'march': '03', 'april': '04', 'may': '05', 'june': '06',
                        'july': '07', 'august': '08', 'september': '09', 'october': '10', 'november': '11', 'december': '12'
                    }
                    month_num = month_map[month_str]
                    date_filter = f"WHERE DATE_FORMAT(p.created_date, '%Y-%m') = '{year_str}-{month_num}'"
                    
                    # Only add if not already present AND no existing WHERE with specific date
                    # ALSO: Don't modify weekly/temporal aggregation SQL that already has proper date filtering
                    # ALSO: Don't modify revenue queries that already have DATE_SUB filters
                    if (date_filter not in sql_query and 
                        'WHERE DATE(' not in sql_query and 
                        f"= '{year_str}-{month_num}" not in sql_query and
                        'WEEK(' not in sql_query and  # Don't modify weekly SQL
                        'DATE_FORMAT(p.created_date, \'%Y-%m\')' not in sql_query and  # Don't modify if already has month filter
                        'DATE_SUB(' not in sql_query and  # Don't modify revenue queries with DATE_SUB
                        'DATE_ADD(' not in sql_query):  # Don't modify revenue queries with DATE_ADD
                        
                        # FIXED: Properly insert date filter without creating duplicate WHERE clauses
                        if 'WHERE' in sql_query:
                            # Add to existing WHERE clause with AND
                            sql_query = re.sub(r'WHERE\s+', f'WHERE {date_filter} AND ', sql_query, count=1)
                        else:
                            # Add new WHERE clause
                            if 'GROUP BY' in sql_query:
                                sql_query = sql_query.replace('GROUP BY', f'WHERE {date_filter} GROUP BY')
                            else:
                                sql_query += f' WHERE {date_filter}'
        
        return sql_query

    def _ensure_closed_quotes(self, sql_query: str) -> str:
        """Ensure all single and double quotes in the SQL query are properly closed."""
        # If odd number of single quotes, append one
        if sql_query.count("'") % 2 != 0:
            sql_query += "'"
        # If odd number of double quotes, append one
        if sql_query.count('"') % 2 != 0:
            sql_query += '"'
        return sql_query

    def _fix_field_selection_issues(self, sql_query: str, user_query: str) -> str:
        user_query_lower = user_query.lower()
        # Fix: User asks for subscription value but SQL uses payment amounts
        if 'subscription' in user_query_lower and 'value' in user_query_lower:
            if 'p.trans_amount_decimal' in sql_query:
                sql_query = sql_query.replace(
                    'p.trans_amount_decimal', 
                    'COALESCE(c.renewal_amount, c.max_amount_decimal, 0)'
                )
        # Add NULL handling
        if 'c.user_email' in sql_query and 'COALESCE' not in sql_query:
            sql_query = sql_query.replace('c.user_email', 'COALESCE(c.user_email, "Not provided")')
        return sql_query

    def _validate_field_usage(self, sql_query: str, user_query: str) -> str:
        """Validate and fix common field usage mistakes"""
        user_lower = user_query.lower()
        # Fix: Revenue query using subscription amounts
        if any(word in user_lower for word in ['revenue', 'money earned', 'payment total']):
            if 'c.renewal_amount' in sql_query and 'p.trans_amount_decimal' not in sql_query:
                logger.warning("🔧 Revenue query should use payment amounts, not subscription amounts")
        # Fix: Subscription value query using payment amounts  
        if any(phrase in user_lower for phrase in ['subscription value', 'subscription amount', 'subscription cost']):
            if 'p.trans_amount_decimal' in sql_query and 'c.renewal_amount' not in sql_query:
                logger.warning("🔧 Subscription value query should use subscription amounts, not payment amounts")
        # Add NULL handling if missing
        if 'c.user_email' in sql_query and 'COALESCE' not in sql_query:
            sql_query = sql_query.replace('c.user_email', 'COALESCE(c.user_email, "Email not provided")')
        if 'c.user_name' in sql_query and 'COALESCE' not in sql_query:
            sql_query = sql_query.replace('c.user_name', 'COALESCE(c.user_name, "Name not provided")')
        return sql_query

    def _fix_column_name_typos(self, sql_query: str) -> str:
        """Fix common column name typos that cause SQL failures"""
        # Fix the most common typo: subscription_start_date -> subcription_start_date
        sql_query = sql_query.replace('subscription_start_date', 'subcription_start_date')
        sql_query = sql_query.replace('subscription_end_date', 'subcription_end_date')
        
        # Ensure proper table aliases and JOINs
        if 'FROM subscription_contract_v2' in sql_query and ' c' not in sql_query:
            sql_query = sql_query.replace('FROM subscription_contract_v2', 'FROM subscription_contract_v2 c')
        if 'FROM subscription_payment_details' in sql_query and ' p' not in sql_query:
            sql_query = sql_query.replace('FROM subscription_payment_details', 'FROM subscription_payment_details p')
            
        return sql_query

    def record_feedback(self, query, feedback):
        self.last_feedback = feedback
        self.last_feedback_query = query

class CompleteEnhancedResultFormatter:
    """COMPLETE enhanced result formatter with smart insights and better presentation."""
    
    def __init__(self):
        self.graph_generator = CompleteGraphGenerator()
    
    def format_single_result(self, result, show_details=False, show_graph=True):
        """Enhanced result formatting with better NULL handling and validation"""
        # ENFORCE: Always show graph if tool_used is execute_dynamic_sql_with_graph or wants_graph is True
        wants_graph = False
        tool_used = None
        data = None
        graph_type = 'bar'
        query = ''
        if hasattr(result, 'tool_used'):
            tool_used = result.tool_used
        if hasattr(result, 'wants_graph'):
            wants_graph = result.wants_graph
        if hasattr(result, 'data'):
            data = result.data
        if hasattr(result, 'parameters') and result.parameters:
            graph_type = result.parameters.get('graph_type', 'bar')
        if hasattr(result, 'original_query'):
            query = result.original_query
        # Also support dicts
        if isinstance(result, dict):
            tool_used = result.get('tool_used', tool_used)
            wants_graph = result.get('wants_graph', wants_graph)
            data = result.get('data', data)
            params = result.get('parameters', {})
            if params:
                graph_type = params.get('graph_type', graph_type)
            query = result.get('original_query', query)
        if (tool_used == 'execute_dynamic_sql_with_graph' or wants_graph) and data and isinstance(data, list) and len(data) > 0:
            graph_data = {
                'data': data,
                'graph_type': graph_type
            }
            graph_path = self.graph_generator.generate_graph(graph_data, query)
            if graph_path:
                print_success(f"\n[Graph generated: {graph_path}]")
        if not result or not isinstance(result, dict):
            return {"error": "No data available"}
            
        formatted_row = {}
        for k, v in result.items():
            if v is None or v == '':
                if 'email' in k.lower():
                    formatted_row[k] = 'Email not provided'
                elif 'name' in k.lower():
                    formatted_row[k] = 'Name not provided'  
                elif any(word in k.lower() for word in ['amount', 'revenue', 'value', 'total']):
                    formatted_row[k] = '0.00'
                else:
                    formatted_row[k] = 'N/A'
            else:
                # Format currency amounts - but NOT count values
                if (any(word in k.lower() for word in ['amount', 'revenue', 'total']) and 
                    not any(word in k.lower() for word in ['count', 'num', 'number', 'qty', 'quantity']) and
                    isinstance(v, (int, float))):
                    try:
                        formatted_row[k] = f"{float(v):,.2f}"
                    except:
                        formatted_row[k] = str(v)
                # Format count values without dollar signs
                elif any(word in k.lower() for word in ['count', 'num', 'number', 'qty', 'quantity']) and isinstance(v, (int, float)):
                    try:
                        formatted_row[k] = f"{int(v):,}"
                    except:
                        formatted_row[k] = str(v)
                # Format 'value' field - check if it's actually a count based on context
                elif 'value' in k.lower() and isinstance(v, (int, float)):
                    # If the value is a whole number and likely a count, don't add dollar sign
                    if float(v).is_integer() and float(v) < 10000:  # Heuristic for count vs currency
                        formatted_row[k] = f"{int(v):,}"
                    else: 
                        formatted_row[k] = f"{float(v):,.2f}"
                else:
                    formatted_row[k] = str(v)
        return formatted_row
    
    def format_result(self, result, show_details=False, show_graph=True):
        """Enhanced result formatting with empty result handling and debug output"""
        print("[DEBUG] Raw result passed to formatter:", repr(result))  # Add debug output
        if result is None:
            return """
======
RESULT
======
Query executed but returned no data.
This could indicate:
- Empty result set
- SQL syntax errors
- Invalid column names or table references

Please check your query and try again.
------------------------------------------------------------
"""
        # If result is a list with at least one dict, show all as table if more than 1 row
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            if len(result) == 1:
                formatted = self.format_single_result(result[0], show_details, show_graph)
                if "error" in formatted:
                    return f"""
======
RESULT
======
{formatted["error"]}
------------------------------------------------------------
"""
                output = "\n======\nRESULT\n======\n"
                for k, v in formatted.items():
                    output += f"{k}: {v}\n"
                # If this is a success/failure breakdown, add rates
                if len(result) == 2 and all('category' in row and 'value' in row for row in result):
                    cats = {row['category'].lower(): row['value'] for row in result}
                    total = sum(cats.values())
                    if total > 0 and any(cat in cats for cat in ['success', 'failure', 'failed']):
                        success = cats.get('success', 0)
                        failure = cats.get('failure', cats.get('failed', 0))
                        output += f"Success Rate: {success / total * 100:.1f}%\n"
                        output += f"Failure Rate: {failure / total * 100:.1f}%\n"
                output += "------------------------------------------------------------\n"
                return output
            else:
                # Table format for multiple rows
                headers = list(result[0].keys())
                output = "\n======\nRESULT\n======\n"
                output += " | ".join(headers) + "\n"
                output += "------" + "-" * (len(" | ".join(headers)) - 6) + "\n"
                for row in result:
                    values = [str(row.get(h, 'N/A')) for h in headers]
                    output += " | ".join(values) + "\n"
                output += "------------------------------------------------------------\n"
                return output
        # If result is a list but empty, show no data
        if isinstance(result, list) and len(result) == 0:
            return """
======
RESULT
======
No data found. This could be due to:
- Invalid date ranges or filters
- No records matching the criteria
- Database connection issues

Try:
- Using broader date ranges
- Checking for typos in your query
- Simplifying your request
------------------------------------------------------------
"""
        # Handle other result types
        return f"""
======
RESULT
======
{str(result)}
------------------------------------------------------------
"""

    def format_multi_result(self, results, query):
        """Format multiple query results."""
        if not results:
            return "No results found."
        
        formatted_outputs = []
        for i, result in enumerate(results, 1):
            if hasattr(result, 'data') and result.data:
                formatted_output = f"Result {i}:\n"
                formatted_output += self.format_result(result.data)
            else:
                formatted_output = f"Result {i}: No data available"
            formatted_outputs.append(formatted_output)
        
        return "\n\n".join(formatted_outputs)

# Complete Enhanced Universal Client with all fixes
class CompleteEnhancedUniversalClient:
    """COMPLETE universal client with full semantic learning and proper schema handling."""
    
    def __init__(self, config: dict):
        self.config = config
        self.nlp = CompleteSmartNLPProcessor()
        self.session = None
        self.formatter = CompleteEnhancedResultFormatter()
        self.history = []
        self.graph_generator = CompleteGraphGenerator()
        self.max_history_length = 8  # Increased for better context
        self.ssl_disabled = False
        self.context = {}  # Store key results for context awareness

    async def __aenter__(self):
        try:
            try:
                connector = aiohttp.TCPConnector(
                    ssl=ssl.create_default_context(),
                    limit=10,
                    limit_per_host=5,
                    enable_cleanup_closed=True,
                    force_close=True,
                    ttl_dns_cache=300
                )
                self.session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=aiohttp.ClientTimeout(total=120),
                    headers={'Connection': 'close'}
                )
                return self
            except Exception as ssl_error:
                print("⚠️ SSL verification failed, retrying with SSL disabled (not secure)...")
                connector = aiohttp.TCPConnector(ssl=False)
                self.session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=aiohttp.ClientTimeout(total=120),
                    headers={'Connection': 'close'}
                )
                self.ssl_disabled = True
                return self
        except Exception as e:
            logger.error(f"Failed to initialize complete client: {e}")
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.session:
                await self.session.close()
        except Exception as e:
            logger.warning(f"Error closing complete session: {e}")

    async def call_tool(self, tool_name: str, parameters: Dict = None, original_query: str = "", wants_graph: bool = False) -> QueryResult:
        """Complete enhanced tool calling with smart graph handling."""
        # ENFORCE: Always apply LIMIT enforcement to the SQL before execution
        if parameters and 'sql_query' in parameters:
            parameters['sql_query'] = self.nlp._enforce_top_n_limit(parameters['sql_query'], original_query)
        if tool_name == 'execute_dynamic_sql_with_graph':
            return await self._handle_complete_smart_sql_with_graph(parameters, original_query)
        
        headers = {
            "Authorization": f"Bearer {self.config['API_KEY_1']}",
            "Connection": "close"
        }
        payload = {"tool_name": tool_name, "parameters": parameters or {}}
        server_url = self.config['SUBSCRIPTION_API_URL']
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                async with self.session.post(f"{server_url}/execute", json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        if attempt < max_retries - 1:
                            logger.warning(f"HTTP {response.status} on attempt {attempt + 1}, retrying...")
                            await asyncio.sleep(1)
                            continue
                        return QueryResult(
                            success=False,
                            error=f"HTTP {response.status}: {error_text}",
                            tool_used=tool_name
                        )
                    result_data = await response.json()
                    print("[DEBUG] Raw API response:", result_data)  # Add debug output
                    # Always set generated_sql to the final SQL (with LIMIT)
                    return QueryResult(
                        success=result_data.get('success', False),
                        data=result_data.get('data'),
                        error=result_data.get('error'),
                        message=result_data.get('message'),
                        tool_used=tool_name,
                        parameters=parameters,
                        is_dynamic=(tool_name == 'execute_dynamic_sql'),
                        original_query=original_query,
                        generated_sql=parameters.get('sql_query') if 'sql_query' in parameters else None
                    )
            except ssl.SSLError as ssl_error:
                if not self.ssl_disabled:
                    print("⚠️ SSL error encountered, retrying with SSL disabled (not secure)...")
                    connector = aiohttp.TCPConnector(ssl=False)
                    self.session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=aiohttp.ClientTimeout(total=120),
                        headers={'Connection': 'close'}
                    )
                    self.ssl_disabled = True
                    continue
                else:
                    return QueryResult(success=False, error=f"SSL error: {ssl_error}", tool_used=tool_name)
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Complete attempt {attempt + 1} failed: {e}, retrying...")
                    await asyncio.sleep(1)
                    continue
                else:
                    return QueryResult(
                        success=False,
                        error=f"Complete connection error after {max_retries} attempts: {str(e)}",
                        tool_used=tool_name
                    )
        
        return QueryResult(
            success=False,
            error="All complete retry attempts failed",
            tool_used=tool_name
        )

    async def _handle_complete_smart_sql_with_graph(self, parameters: Dict, original_query: str) -> QueryResult:
        """FIXED: Complete smart SQL with graph generation handling."""
        try:
            # ENFORCE: Always apply LIMIT enforcement to the SQL before execution
            if parameters and 'sql_query' in parameters:
                parameters['sql_query'] = self.nlp._enforce_top_n_limit(parameters['sql_query'], original_query)
            # Execute SQL first
            sql_result = await self.call_tool('execute_dynamic_sql', {
                'sql_query': parameters['sql_query']
            }, original_query)
            # Always set generated_sql to the final SQL (with LIMIT)
            sql_result.generated_sql = parameters['sql_query']
            if not sql_result.success or not sql_result.data:
                if sql_result.success:
                    sql_result.message = (sql_result.message or "") + "\n💡 No data returned - cannot generate complete graph"
                return sql_result
            logger.info(f"📊 Complete SQL returned {len(sql_result.data)} rows for graph analysis")
            if not self.graph_generator.can_generate_graphs():
                sql_result.message = (sql_result.message or "") + "\n⚠️ Complete graph generation unavailable (matplotlib not installed)"
                return sql_result
            # Generate graph with complete enhanced handling
            try:
                graph_data = {
                    'data': sql_result.data,  # Pass SQL result data
                    'graph_type': parameters.get('graph_type', 'bar'),
                    'title': self._generate_complete_smart_title(original_query),
                    'description': f"Generated from: {original_query}"
                }
                enforced_type = parameters.get('graph_type', 'auto')
                if enforced_type == 'auto':
                    query_lower = original_query.lower()
                    if any(word in query_lower for word in ['trend', 'trends', 'over time', 'timeline', 'payment trends']):
                        enforced_type = 'line'
                        logger.info(f"[SMART-DETECT] Detected trend query, using line chart for: {original_query}")
                    elif any(word in query_lower for word in ['pie', 'distribution', 'breakdown']):
                        enforced_type = 'pie'
                        logger.info(f"[SMART-DETECT] Detected distribution query, using pie chart")
                    else:
                        enforced_type = 'bar'  # Default fallback
                        logger.info(f"[SMART-DETECT] Using default bar chart")
                graph_data['graph_type'] = enforced_type
                logger.info(f"[ENFORCE] Setting graph_data['graph_type'] = '{enforced_type}' (from params: {parameters.get('graph_type', 'not_set')}) for query: {original_query[:50]}...")
                graph_filepath = self.graph_generator.generate_graph(
                    graph_data, original_query
                )
                sql_result.graph_data = graph_data
                sql_result.graph_generated = graph_filepath is not None
                if graph_filepath:
                    sql_result.graph_filepath = graph_filepath
                    sql_result.message = (sql_result.message or "") + f"\n📊 Complete graph generated successfully"
                else:
                    sql_result.message = (sql_result.message or "") + f"\n⚠️ Complete graph data generated but file creation failed"
            except Exception as graph_error:
                logger.error(f"Complete graph generation error: {graph_error}")
                sql_result.message = (sql_result.message or "") + f"\n⚠️ Complete graph generation failed: {str(graph_error)}"
            return sql_result
        except Exception as e:
            logger.error(f"Error in complete smart SQL with graph: {e}")
            # Fallback to regular SQL
            return await self.call_tool('execute_dynamic_sql', {
                'sql_query': parameters['sql_query']
            }, original_query)

    def _generate_complete_smart_title(self, query: str) -> str:
        """Generate complete smart title from query."""
        try:
            query_lower = query.lower()
            
            if 'success' in query_lower and 'rate' in query_lower:
                return "Complete Payment Success Analysis"
            elif 'merchant' in query_lower and 'transaction' in query_lower:
                return "Complete Merchant Transaction Analysis"
            elif 'trend' in query_lower:
                return "Complete Trend Analysis"
            elif 'pie' in query_lower or 'distribution' in query_lower:
                return "Complete Distribution Analysis"
            elif 'payment' in query_lower:
                return "Complete Payment Analysis"
            elif 'subscription' in query_lower:
                return "Complete Subscription Analysis"
            else:
                return "Complete Data Analysis"
        except Exception:
            return "Complete Analysis"

    async def query(self, user_query: str) -> Union[QueryResult, List[QueryResult]]:
        """Complete enhanced query processing with smart AI and MULTITOOL support. Includes post-processing for metric queries."""
        try:
            parsed_calls = await self.nlp.parse_query(user_query, self.history, client=self)
            
            if len(parsed_calls) > 1:
                results = []
                for call in parsed_calls:
                    try:
                        result = await self.call_tool(
                            call['tool'], 
                            call['parameters'], 
                            call['original_query'],
                            call.get('wants_graph', False)
                        )
                        # Add multitool metadata
                        result.query_index = call.get('query_index', 1)
                        result.total_queries = call.get('total_queries', len(parsed_calls))
                        result.is_multitool = call.get('is_multitool', True)
                        # Store SQL queries in context for "visualize that" functionality
                        if call['tool'] in ['execute_dynamic_sql', 'execute_dynamic_sql_with_graph']:
                            sql_query = call['parameters'].get('sql_query')
                            if sql_query:
                                self.context['last_sql_query'] = sql_query
                                logger.info(f"[CONTEXT] Stored SQL query: {sql_query[:100]}...")
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error calling complete tool {call['tool']}: {e}")
                        error_result = QueryResult(
                            success=False,
                            error=f"Complete tool {call['tool']} failed: {str(e)}",
                            tool_used=call['tool']
                        )
                        error_result.query_index = call.get('query_index', 1)
                        error_result.total_queries = call.get('total_queries', len(parsed_calls))
                        error_result.is_multitool = call.get('is_multitool', True)
                        results.append(error_result)
                
                # Post-process for metric queries: if user asked for ARPU/metric and got a breakdown, retry with explicit prompt
                metric_keywords = ['arpu', 'average revenue per user', 'average revenue', 'mean revenue', 'arppu', 'arpau', 'total revenue', 'sum', 'average', 'mean']
                user_query_lower = user_query.lower()
                is_metric_query = any(k in user_query_lower for k in metric_keywords)
                
                if is_metric_query:
                    for result in results:
                        if result.data and isinstance(result.data, list) and len(result.data) > 1 and any('category' in row for row in result.data if isinstance(row, dict)):
                            logger.info("Detected breakdown for metric query; retrying with explicit single-value prompt.")
                            explicit_query = user_query + " (Return only a single ARPU value, not a breakdown or category table.)"
                            return await self.query(explicit_query)
                
                return results
            else:
                call = parsed_calls[0]
                result = await self.call_tool(
                    call['tool'], 
                    call['parameters'], 
                    call['original_query'],
                    call.get('wants_graph', False)
                )
                # Add single query metadata
                result.query_index = 1
                result.total_queries = 1
                result.is_multitool = False
                # Store SQL queries in context for "visualize that" functionality
                if call['tool'] in ['execute_dynamic_sql', 'execute_dynamic_sql_with_graph']:
                    sql_query = call['parameters'].get('sql_query')
                    if sql_query:
                        self.context['last_sql_query'] = sql_query
                        logger.info(f"[CONTEXT] Stored SQL query: {sql_query[:100]}...")
                
                # Post-process for metric queries: if user asked for ARPU/metric and got a breakdown, retry with explicit prompt
                metric_keywords = ['arpu', 'average revenue per user', 'average revenue', 'mean revenue', 'arppu', 'arpau', 'total revenue', 'sum', 'average', 'mean']
                user_query_lower = user_query.lower()
                is_metric_query = any(k in user_query_lower for k in metric_keywords)
                
                if is_metric_query:
                    if result.data and isinstance(result.data, list) and len(result.data) > 1 and any('category' in row for row in result.data if isinstance(row, dict)):
                        logger.info("Detected breakdown for metric query; retrying with explicit single-value prompt.")
                        explicit_query = user_query + " (Return only a single ARPU value, not a breakdown or category table.)"
                        return await self.query(explicit_query)
                
                return result
        except Exception as e:
            logger.error(f"Error in complete query processing: {e}", exc_info=True)
            return QueryResult(
                success=False,
                error=f"Complete query processing failed: {e}",
                tool_used="complete_query_processor"
            )

    def manage_history(self, query: str, response: str):
        """Complete enhanced history management with smart filtering and SQL tracking."""
        try:
            sql_match = re.search(r'(SELECT.*?FROM subscription_[^;]*)', response, re.IGNORECASE | re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1)
                self.context['last_sql_query'] = sql_query
                logger.info(f"[HISTORY] Stored SQL query for context: {sql_query[:100]}...")
            self.history.extend([f"User: {query}", f"Assistant: {response[:200]}..."])
            self.history = self.history[-self.max_history_length:]
        except Exception as e:
            logger.warning(f"Error managing complete history: {e}")

    async def submit_feedback(self, result: QueryResult, helpful: bool, improvement_suggestion: str = None):
        """Complete enhanced feedback submission with better error handling."""
        if result.is_dynamic and result.generated_sql and result.original_query:
            try:
                feedback_params = {
                    'original_question': result.original_query,
                    'sql_query': result.generated_sql,
                    'was_helpful': helpful
                }
                
                if not helpful and improvement_suggestion:
                    feedback_params['improvement_suggestion'] = improvement_suggestion.strip()
                
                feedback_result = await self.call_tool('record_query_feedback', feedback_params)
                
                if feedback_result.success and feedback_result.message:
                    print(f"✅ {feedback_result.message}")
                else:
                    print(f"✅ Complete feedback recorded successfully")
                    
            except Exception as e:
                logger.warning(f"Could not submit complete feedback: {str(e)}")
                print("⚠️ Complete feedback noted locally")

# Complete Enhanced Interactive Mode
async def complete_enhanced_interactive_mode():
    """COMPLETE interactive mode with full functionality."""
    print("✨ COMPLETE Subscription Analytics AI Agent ✨")
    print("=" * 70)
    
    config_manager = ConfigManager()
    
    try:
        user_config = config_manager.get_config()
        genai.configure(api_key=user_config['GOOGLE_API_KEY'])
        
        print(f"🔗 Connected to server: {user_config['SUBSCRIPTION_API_URL']}")
        print("🧠 COMPLETE Smart AI with full semantic learning!")
        print("🛡️ Production-ready error handling and recovery!")
        print("🔧 COMPLETE SQL generation with perfect schema handling!")
        print("📊 COMPLETE semantic learning with feedback system!")
        print("🎯 COMPLETE chart type awareness and learning!")
        print("🔗 MULTITOOL functionality for complex queries!")
        
        if MATPLOTLIB_AVAILABLE:
            print("📈 COMPLETE advanced graph generation with smart type detection")
        else:
            print("⚠️ Graph generation disabled (install matplotlib: pip install matplotlib)")
        
        async with CompleteEnhancedUniversalClient(config=user_config) as client:
            print("\n💬 Enter questions in natural language. Type 'quit' to exit.")
            print("\n📚 COMPLETE Examples:")
            print("  • Show me a pie chart of payment success rates")
            print("  • Give me subscription trends as a line chart")
            print("  • Visualize payment data with a bar chart")
            print("  • Create a pie chart breakdown of successful vs failed payments")
            print("  • Show payment success rate for merchants with more than 1 transaction visually")
            print("  • Show payment trends over time")
            print("  • Tell me the last date for which data is available")
            print("  • Compare subscribers with more than 1 and more than 2 subscriptions")
            print("  • Multiple queries: Get database status and show recent payment summary")
            print("\n💡 The COMPLETE AI has semantic learning and MULTITOOL support!")
            print("=" * 70)
            
            while True:
                try:
                    query = input("\n> ").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        break
                    if not query:
                        continue
                    
                    print("🤔 Processing your query with COMPLETE AI...")
                    result = await client.query(query)
                    
                    try:
                        if isinstance(result, list):
                            output = client.formatter.format_multi_result(result, query)
                        else:
                            # Use format_result for QueryResult objects, format_single_result is for individual data rows
                            output = client.formatter.format_result(result.data if hasattr(result, 'data') else result, show_details=args.show_details)
                        
                        print(f"\n{output}")
                        
                        client.manage_history(query, output)
                        
                        # COMPLETE enhanced feedback for dynamic queries
                        if isinstance(result, list):
                            # Handle multitool feedback
                            for i, individual_result in enumerate(result, 1):
                                if (individual_result.is_dynamic and 
                                    individual_result.success and 
                                    individual_result.data is not None):
                                    
                                    print(f"\n" + "="*50)
                                    print(f"📝 Result {i} was generated using COMPLETE AI with semantic learning!")
                                    print("Your feedback helps the system learn and improve over time.")
                                    
                                    last_query = query
                                    last_result = individual_result
                                    last_history = list(client.history) if hasattr(client, 'history') else []
                                    while True:
                                        try:
                                            feedback_input = input(f"Was Result {i} helpful? (y/n/skip): ").lower().strip()
                                            if feedback_input in ['y', 'yes']:
                                                await client.submit_feedback(last_result, True)
                                                print("🧠 Positive feedback recorded in semantic learning system!")
                                                print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                                break
                                            elif feedback_input in ['n', 'no']:
                                                improvement = input("How can this be improved? (e.g., 'use pie chart instead', 'fix SQL error'): ").strip()
                                                if improvement.lower() not in ['skip', 's', '']:
                                                    await client.submit_feedback(last_result, False, improvement)
                                                    print("🧠 Negative feedback and improvement recorded - the system will learn!")
                                                    print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                                    # PATCH: Inject feedback into history for immediate retry
                                                    last_history.append(f"How can this be improved? {improvement}")
                                                else:
                                                    await client.submit_feedback(last_result, False)
                                                    print("🧠 Negative feedback recorded!")
                                                    print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                                    # PATCH: Still inject a generic feedback line for retry
                                                    last_history.append("How can this be improved? (no details)")
                                                # PATCH: Re-run the last query after negative feedback
                                                print(f"\n🔄 Regenerating answer for Result {i} based on your feedback...\n")
                                                last_result = await client.query(last_query, history=last_history)
                                                # For multitool, get the i-th result again if possible
                                                if isinstance(last_result, list) and len(last_result) >= i:
                                                    new_individual_result = last_result[i-1]
                                                else:
                                                    new_individual_result = last_result if hasattr(last_result, 'data') else last_result
                                                if hasattr(new_individual_result, 'data'):
                                                    output = client.formatter.format_result(new_individual_result.data, show_details=args.show_details)
                                                else:
                                                    output = client.formatter.format_result(new_individual_result, show_details=args.show_details)
                                                print(f"\n{output}")
                                                last_result = new_individual_result
                                                continue  # Ask for feedback again
                                            elif feedback_input in ['s', 'skip', '']:
                                                break
                                            else:
                                                print("Please enter 'y', 'n', or 'skip'.")
                                        except (KeyboardInterrupt, EOFError):
                                            break
                        else:
                            # Handle single result feedback
                            if (result.is_dynamic and 
                                result.success and 
                                result.data is not None):
                                
                                print("\n" + "="*50)
                                print("📝 This answer was generated using COMPLETE AI with semantic learning!")
                                print("Your feedback helps the system learn and improve over time.")
                                
                                last_query = query
                                last_result = result
                                last_history = list(client.history) if hasattr(client, 'history') else []
                                while True:
                                    try:
                                        feedback_input = input("Was this helpful? (y/n/skip): ").lower().strip()
                                        if feedback_input in ['y', 'yes']:
                                            await client.submit_feedback(last_result, True)
                                            print("🧠 Positive feedback recorded in semantic learning system!")
                                            print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                            break
                                        elif feedback_input in ['n', 'no']:
                                            improvement = input("How can this be improved? (e.g., 'use pie chart instead', 'fix SQL error'): ").strip()
                                            if improvement.lower() not in ['skip', 's', '']:
                                                await client.submit_feedback(last_result, False, improvement)
                                                print("🧠 Negative feedback and improvement recorded - the system will learn!")
                                                print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                                # PATCH: Inject feedback into history for immediate retry
                                                last_history.append(f"How can this be improved? {improvement}")
                                            else:
                                                await client.submit_feedback(last_result, False)
                                                print("🧠 Negative feedback recorded!")
                                                print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                                # PATCH: Still inject a generic feedback line for retry
                                                last_history.append("How can this be improved? (no details)")
                                            # PATCH: Re-run the last query after negative feedback
                                            print("\n🔄 Regenerating answer based on your feedback...\n")
                                            last_result = await client.query(last_query, history=last_history)
                                            if hasattr(last_result, 'data'):
                                                output = client.formatter.format_result(last_result.data, show_details=args.show_details)
                                            else:
                                                output = client.formatter.format_result(last_result, show_details=args.show_details)
                                            print(f"\n{output}")
                                            continue  # Ask for feedback again
                                        elif feedback_input in ['s', 'skip', '']:
                                            break
                                        else:
                                            print("Please enter 'y', 'n', or 'skip'.")
                                    except (KeyboardInterrupt, EOFError):
                                        break
                                # If feedback was negative, loop will repeat and regenerate improved answer
                                if feedback_input in ['n', 'no']:
                                    continue
                                else:
                                    break
                    except Exception as format_error:
                        logger.error(f"Error formatting COMPLETE output: {format_error}")
                        print(f"❌ Error displaying COMPLETE results: {format_error}")
                        if isinstance(result, QueryResult):
                            print(f"Raw result: Success={result.success}, Error={result.error}")
                        elif isinstance(result, list):
                            print(f"Raw results: {len(result)} results, {sum(1 for r in result if r.success)} successful")
                
                except (KeyboardInterrupt, EOFError):
                    break
                except Exception as e:
                    logger.error("Error in COMPLETE interactive loop", exc_info=True)
                    print(f"❌ Error: {e}")
                    print("💡 The COMPLETE system will continue - try a different query")
                    
    except Exception as e:  
        logger.error("COMPLETE client failed to initialize", exc_info=True)
        print(f"❌ Critical Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify your API keys in config.json")
        print("3. Ensure the server is running")
        if not MATPLOTLIB_AVAILABLE:
            print("4. For COMPLETE graphs: pip install matplotlib")
        print("5. For COMPLETE semantic learning: pip install sentence-transformers faiss-cpu")
    
    print("\n👋 Goodbye from the COMPLETE system!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subscription Analytics Universal Client")
    parser.add_argument('query', nargs='?', help='Natural language query to run')
    parser.add_argument('--test-connection', action='store_true', help='Test API server connection and credentials')
    parser.add_argument('--show-config', action='store_true', help='Show current config (without API keys)')
    parser.add_argument('--show-server-status', action='store_true', help='Show current server status')
    parser.add_argument('--show-gemini-key', action='store_true', help='Show Gemini API key status (masked)')
    parser.add_argument('--show-details', action='store_true', help='Show technical/log details (SQL, tool info, etc.)')
    args = parser.parse_args()
    
    config_manager = ConfigManager()
    config = config_manager.get_config()
    print(f"✅ Config loaded from: {config_manager.config_path}")
    print(f"✅ Server URL: {config.get('SUBSCRIPTION_API_URL', '[not set]')}")
    
    # Gemini API key check
    gemini_key = config.get('GOOGLE_API_KEY', '')
    if not gemini_key or not isinstance(gemini_key, str) or len(gemini_key) < 20 or not gemini_key.startswith('AI'):
        print("❌ Gemini API key is missing or invalid!")
        print("   Please set it in client/config.json as 'GOOGLE_API_KEY' or export GOOGLE_API_KEY=your_key")
        print("   Get your key from https://ai.google.dev/")
        sys.exit(1)
    
    import google.generativeai as genai
    try:
        genai.configure(api_key=gemini_key)
    except Exception as e:
        print(f"❌ Failed to configure Gemini API: {e}")
        sys.exit(1)
    
    if args.show_config:
        safe_config = {k: ('***' if 'key' in k.lower() else v) for k, v in config.items()}
        print(json.dumps(safe_config, indent=2))
        sys.exit(0)
    
    if args.show_server_status:
        print("Checking server status...")
        import requests
        try:
            resp = requests.get(config['SUBSCRIPTION_API_URL'] + '/health', timeout=10)
            print(f"Server status: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"❌ Could not reach server: {e}")
        sys.exit(0)
    
    if args.show_gemini_key:
        print(f"Gemini API key: {'*' * (len(gemini_key) - 4) + gemini_key[-4:] if gemini_key else '[not set]'}")
        sys.exit(0)
    
    if args.test_connection:
        print("Testing API connection...")
        import requests
        try:
            resp = requests.get(config['SUBSCRIPTION_API_URL'] + '/health', timeout=10)
            if resp.status_code == 200:
                print("✅ API server is reachable and healthy.")
            else:
                print(f"❌ API server returned status {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"❌ Could not connect to API server: {e}")
        sys.exit(0)

    EXAMPLE_QUERIES = [
        "Visualize payment data with a bar chart",
        "Create a pie chart breakdown of successful vs failed payments",
        "Show payment success rate for merchants with more than 1 transaction visually",
        "number of merchants with more than 5 subscriptions and number of merchants with more than 5 payments",
        "Show payment trends over time",
        "Tell me the last date for which data is available",
        "Compare subscribers with more than 1 and more than 2 subscriptions",
        "Number of subscriptions on 24 april 2025",
        "Revenue for 24 april 2025",
        "Number of subscriptions between 1 may 2025 and 31 may 2025",
        "Revenue between 1 april 2025 and 30 april 2025",
        "Show me database status and recent subscription summary",
        "How many new subscriptions did we get this month?",
        "Show me a pie chart of payment success rates and show me a bar chart of the top 5 merchants by total payment revenue",
        "Show me users with their email addresses and subscription amounts",
        "Show me the top 10 customers by total subscription value",
    ]

    def print_example_queries():
        print("\n💡 Example queries:")
        for q in EXAMPLE_QUERIES:
            print(f"  • {q}")

    async def run_query_loop():
        print_header("✨ COMPLETE Subscription Analytics AI Agent ✨")
        print_section("Welcome! Type your questions below. Type 'exit' to quit.")
        async with CompleteEnhancedUniversalClient(config) as client:
            while True:
                try:
                    user_query = args.query
                    if not user_query:
                        print_section("Example queries:")
                        for q in EXAMPLE_QUERIES:
                            print(f"  {CYAN}•{RESET} {q}")
                        print_separator()
                        try:
                            user_query = input(f"{BOLD}{YELLOW}\nEnter your query (or 'exit' to quit): {RESET}").strip()
                        except (KeyboardInterrupt, EOFError):
                            print_success("\n👋 Goodbye from COMPLETE system!")
                            break
                        if user_query.lower() in ['exit', 'quit', 'q']:
                            print_success("\n👋 Goodbye from COMPLETE system!")
                            break
                        if not user_query:
                            continue
                    # Always reload improvement suggestions before each query for immediate feedback effect
                    if hasattr(client.nlp, '_last_best_chart_type'):
                        del client.nlp._last_best_chart_type
                    # Context-aware query resolution
                    resolved_query = user_query
                    if 'that day' in user_query.lower() and 'last_date' in client.context:
                        resolved_query = user_query.lower().replace('that day', client.context['last_date'])
                    if 'that merchant' in user_query.lower() and 'last_merchant' in client.context:
                        resolved_query = user_query.lower().replace('that merchant', client.context['last_merchant'])
                    if 'that count' in user_query.lower() and 'last_count' in client.context:
                        resolved_query = user_query.lower().replace('that count', str(client.context['last_count']))

                    # --- RECURSIVE FEEDBACK LOOP START ---
                    def update_context_from_result(result):
                        if result.data and isinstance(result.data, list) and len(result.data) > 0:
                            row = result.data[0]
                            if isinstance(row, dict):
                                for k, v in row.items():
                                    if 'date' in k.lower():
                                        client.context['last_date'] = str(v)
                                    if 'merchant' in k.lower():
                                        client.context['last_merchant'] = str(v)
                                    if 'count' in k.lower() or 'num' in k.lower():
                                        client.context['last_count'] = v

                    async def handle_query_with_feedback(query):
                        feedback = None
                        improvement = None
                        current_query = query
                        while True:
                            # If improvement suggestion exists, append it to the query
                            query_to_run = current_query
                            if improvement:
                                query_to_run = f"{current_query} ({improvement})"
                            result = await client.query(query_to_run)
                            print_separator()
                            # Format and display results
                            if isinstance(result, list):
                                print_header(f"MULTITOOL RESULTS FOR: '{query_to_run}'")
                                output = client.formatter.format_multi_result(result, query_to_run)
                                print(f"{output}")
                                print_separator()
                                for individual_result in result:
                                    update_context_from_result(individual_result)
                                # Feedback for multitool results (recursive)
                                all_satisfied = True
                                for i, individual_result in enumerate(result, 1):
                                    if (getattr(individual_result, 'is_dynamic', False) and 
                                        getattr(individual_result, 'success', False) and 
                                        getattr(individual_result, 'data', None) is not None):
                                        print_feedback_prompt(f"\n📝 Feedback for Query {i} (generated by AI):")
                                        print_feedback_prompt("Your feedback helps the system learn and improve over time.")
                                        while True:
                                            try:
                                                feedback_input = input(f"{MAGENTA}Was Query {i} helpful? (y/n/skip): {RESET}").lower().strip()
                                                if feedback_input in ['y', 'yes', 's', 'skip', '']:
                                                    if feedback_input in ['y', 'yes']:
                                                        await client.submit_feedback(individual_result, True)
                                                        print_success("🧠 Positive feedback recorded in semantic learning system!")
                                                        print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                                    all_satisfied = all_satisfied and True
                                                    improvement = None  # Reset improvement on positive/skip
                                                    break
                                                elif feedback_input in ['n', 'no']:
                                                    improvement = input(f"{MAGENTA}How can this be improved? (e.g., 'use bar chart instead', 'fix SQL error'): {RESET}").strip()
                                                    await client.submit_feedback(individual_result, False, improvement)
                                                    print_success("🧠 Negative feedback and improvement recorded - the system will learn!")
                                                    print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                                    all_satisfied = False
                                                    break
                                                else:
                                                    print_warning("Please enter 'y', 'n', or 'skip'.")
                                            except (KeyboardInterrupt, EOFError):
                                                all_satisfied = True
                                                break
                                if all_satisfied:
                                    break
                                else:
                                    print_section("Regenerating improved answer(s) based on your feedback...")
                                    continue  # Regenerate improved answer(s)
                            else:
                                output = client.formatter.format_result(result.data if hasattr(result, 'data') else result, show_details=args.show_details)
                                print(f"{output}")
                                print_separator()
                                update_context_from_result(result)
                                print_feedback_prompt("\n📝 This answer was generated using COMPLETE AI with semantic learning!")
                                print_feedback_prompt("Your feedback helps the system learn and improve over time.")
                                while True:
                                    try:
                                        feedback_input = input(f"{MAGENTA}Was this helpful? (y/n/skip): {RESET}").lower().strip()
                                        if feedback_input in ['y', 'yes', 's', 'skip', '']:
                                            if feedback_input in ['y', 'yes']:
                                                await client.submit_feedback(result, True)
                                                print("🧠 Positive feedback recorded in semantic learning system!")
                                                print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                            improvement = None  # Reset improvement on positive/skip
                                            break
                                        elif feedback_input in ['n', 'no']:
                                            improvement = input(f"{MAGENTA}How can this be improved? (e.g., 'use pie chart instead', 'fix SQL error'): {RESET}").strip()
                                            await client.submit_feedback(result, False, improvement)
                                            print("🧠 Negative feedback and improvement recorded - the system will learn!")
                                            print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                            print_section("Regenerating improved answer based on your feedback...")
                                            break  # Regenerate improved answer
                                        else:
                                            print("Please enter 'y', 'n', or 'skip'.")
                                    except (KeyboardInterrupt, EOFError):
                                        break
                                # If feedback was negative, loop will repeat and regenerate improved answer
                                if feedback_input in ['n', 'no']:
                                    continue
                                else:
                                    break
                        client.manage_history(query_to_run, output)
                    # --- RECURSIVE FEEDBACK LOOP END ---

                    await handle_query_with_feedback(resolved_query)
                except Exception as e:
                    print_error(f"❌ Error running query: {e}")
                    logger.error(f"Query error: {e}", exc_info=True)
                args.query = None  # After first run, always prompt interactively

    # Run interactive mode or single query
    import asyncio
    asyncio.run(run_query_loop())
