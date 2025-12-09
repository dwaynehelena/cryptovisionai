#!/usr/bin/env python3
"""
Logo Generator - Creates a custom SVG logo for CryptoVisionAI
"""

import os
import math
import random
from typing import Tuple, List
import argparse

def generate_svg_logo(output_path: str, color_scheme: str = "blue") -> None:
    """
    Generate a custom SVG logo for CryptoVisionAI
    
    Args:
        output_path: Path to save the SVG file
        color_scheme: Color scheme to use ('blue', 'green', 'purple')
    """
    # Define color schemes
    color_schemes = {
        "blue": {
            "primary": "#1E88E5",
            "secondary": "#0D47A1",
            "highlight": "#64B5F6",
            "accent": "#E3F2FD"
        },
        "green": {
            "primary": "#43A047",
            "secondary": "#1B5E20",
            "highlight": "#81C784",
            "accent": "#E8F5E9"
        },
        "purple": {
            "primary": "#8E24AA",
            "secondary": "#4A148C",
            "highlight": "#BA68C8",
            "accent": "#F3E5F5"
        }
    }
    
    # Use default scheme if specified one doesn't exist
    colors = color_schemes.get(color_scheme, color_schemes["blue"])
    
    # Set dimensions
    width = 512
    height = 512
    padding = 40
    
    # Start SVG content
    svg_content = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="{colors['primary']}" />
            <stop offset="100%" stop-color="{colors['secondary']}" />
        </linearGradient>
        <linearGradient id="gradient2" x1="0%" y1="100%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="{colors['highlight']}" />
            <stop offset="100%" stop-color="{colors['primary']}" />
        </linearGradient>
        <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur in="SourceAlpha" stdDeviation="10" />
            <feOffset dx="3" dy="3" result="offsetblur" />
            <feComponentTransfer>
                <feFuncA type="linear" slope="0.3" />
            </feComponentTransfer>
            <feMerge>
                <feMergeNode />
                <feMergeNode in="SourceGraphic" />
            </feMerge>
        </filter>
    </defs>
    
    <!-- Background Circle -->
    <circle cx="{width/2}" cy="{height/2}" r="{(width-padding*2)/2}" fill="url(#gradient1)" filter="url(#shadow)" />
    
    <!-- Crypto Graph Lines -->
    <path d="M {padding} {height*0.7} 
             C {width*0.2} {height*0.4}, {width*0.3} {height*0.8}, {width*0.4} {height*0.3} 
             S {width*0.6} {height*0.6}, {width*0.7} {height*0.4} 
             S {width*0.8} {height*0.5}, {width-padding} {height*0.2}"
          stroke="{colors['accent']}" stroke-width="6" fill="none" stroke-linecap="round" />
    
    <!-- AI Network Nodes -->
"""

    # Generate random nodes to represent the neural network
    num_nodes = 8
    nodes = []
    center_x, center_y = width/2, height/2
    radius = (width-padding*3)/3
    
    # Generate node positions in a circular pattern
    for i in range(num_nodes):
        angle = 2 * math.pi * i / num_nodes
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        size = random.uniform(8, 16)
        nodes.append((x, y, size))

    # Add nodes to SVG
    for x, y, size in nodes:
        svg_content += f'    <circle cx="{x}" cy="{y}" r="{size}" fill="{colors["highlight"]}" />\n'
    
    # Connect nodes with lines
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            # Only connect some nodes (not all-to-all)
            if random.random() < 0.4:  # 40% chance to connect
                x1, y1, _ = nodes[i]
                x2, y2, _ = nodes[j]
                opacity = random.uniform(0.2, 0.7)
                svg_content += f'    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colors["accent"]}" stroke-width="2" stroke-opacity="{opacity}" />\n'

    # Add Eye Symbol in the center
    svg_content += f"""
    <!-- Vision/Eye Symbol -->
    <circle cx="{width/2}" cy="{height/2}" r="{width/8}" fill="url(#gradient2)" />
    <circle cx="{width/2}" cy="{height/2}" r="{width/18}" fill="{colors['secondary']}" />
    <circle cx="{width/2-width/32}" cy="{height/2-height/32}" r="{width/48}" fill="white" />
    
    <!-- Text -->
    <text x="{width/2}" y="{height-padding}" font-family="Arial, sans-serif" font-size="40" 
          font-weight="bold" text-anchor="middle" fill="{colors['accent']}">
        CryptoVisionAI
    </text>
</svg>
"""

    # Write SVG file
    with open(output_path, "w") as f:
        f.write(svg_content)
    
    print(f"Generated logo at {output_path}")

def png_from_svg(svg_path: str, png_path: str, size: int = 512) -> bool:
    """
    Convert the SVG to PNG if cairosvg is available
    
    Args:
        svg_path: Path to SVG file
        png_path: Path to output PNG file
        size: Width/height of the PNG
        
    Returns:
        bool: True if conversion succeeded
    """
    try:
        import cairosvg
        cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=size, output_height=size)
        print(f"Converted SVG to PNG at {png_path}")
        return True
    except ImportError:
        print("cairosvg not installed. Install with: pip install cairosvg")
        print(f"SVG file is available at {svg_path}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate custom logo for CryptoVisionAI")
    parser.add_argument("--output", type=str, default="src/visualization/assets/logo.svg", 
                       help="Output path for the SVG file")
    parser.add_argument("--color", type=str, default="blue", choices=["blue", "green", "purple"],
                       help="Color scheme to use")
    parser.add_argument("--png", action="store_true", help="Also generate PNG version")
    parser.add_argument("--size", type=int, default=512, help="Size of the PNG output")
    
    args = parser.parse_args()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate SVG logo
    generate_svg_logo(args.output, args.color)
    
    # Generate PNG if requested
    if args.png:
        png_path = args.output.rsplit(".", 1)[0] + ".png"
        png_from_svg(args.output, png_path, args.size)