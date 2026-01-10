#!/usr/bin/env python3
"""
Script para visualizar resultados de experimentos de sistemas de memoria.
Genera gráficas combinadas (globales + por tipo de pregunta) por evaluador.
"""

import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, Any
import pandas as pd

# Configurar estilo de gráficas
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_results(filepath: str) -> Dict[str, Any]:
    """Carga el archivo JSON de resultados."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_per_type_metrics(evaluator_data: Dict[str, Any]) -> pd.DataFrame:
    """Extrae métricas por tipo de pregunta si están disponibles."""
    if 'per_type_metrics' not in evaluator_data:
        return None
    
    per_type = evaluator_data['per_type_metrics']
    rows = []
    for qtype, metrics in per_type.items():
        row = {'question_type': qtype}
        row.update(metrics)
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_evaluator_metrics(results: Dict[str, Any], output_dir: Path):
    """Genera una gráfica completa por evaluador con métricas globales + por tipo."""
    evaluators = results.get('evaluators', {})
    
    # Obtener todos los tipos de pregunta únicos del dataset para orden consistente
    dataset_meta = results.get('dataset_metadata', {})
    all_question_types = list(dataset_meta.get('question_types', {}).keys())
    
    # Si no hay metadata, extraer de los evaluadores
    if not all_question_types:
        for eval_data in evaluators.values():
            df = extract_per_type_metrics(eval_data)
            if df is not None and not df.empty:
                all_question_types = sorted(df['question_type'].unique().tolist())
                break
    
    # Crear mapeo de colores consistente para cada tipo de pregunta
    color_palette = sns.color_palette("husl", len(all_question_types))
    type_colors = {qtype: color_palette[i] for i, qtype in enumerate(all_question_types)}
    
    for eval_name, eval_data in evaluators.items():
        # Obtener métricas globales
        overall = eval_data.get('overall_metrics', {})
        
        # Filtrar métricas globales relevantes (numéricas, no totales, no conteos de preguntas)
        overall_metrics = {
            k: v for k, v in overall.items() 
            if isinstance(v, (int, float)) 
            and not k.startswith('total_')
            and k not in ['evaluable_questions', 'non_evaluable_questions', 'total_questions', 'count']
        }
        
        # Obtener métricas por tipo
        df_per_type = extract_per_type_metrics(eval_data)
        
        # Si no hay datos, saltar
        if not overall_metrics and (df_per_type is None or df_per_type.empty):
            continue
        
        # Determinar qué métricas mostrar (las que aparecen tanto en global como en per_type)
        if df_per_type is not None and not df_per_type.empty:
            available_metrics = [
                col for col in df_per_type.columns 
                if col != 'question_type' 
                and not col.startswith('total_')
                and col not in ['count']
                and df_per_type[col].dtype in ['float64', 'int64']
                and col in overall_metrics
            ]
        else:
            available_metrics = list(overall_metrics.keys())
        
        if not available_metrics:
            continue
        
        # Crear subplots
        n_metrics = len(available_metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), squeeze=False)
        
        for idx, metric in enumerate(available_metrics):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Preparar datos
            if df_per_type is not None and not df_per_type.empty and metric in df_per_type.columns:
                # Combinar global + por tipo
                type_data = df_per_type[['question_type', metric]].copy()
                
                # Añadir fila global al principio
                all_data = pd.concat([
                    pd.DataFrame({'question_type': ['GLOBAL'], metric: [overall_metrics[metric]]}),
                    type_data
                ], ignore_index=True)
                
                # Ordenar usando el orden fijo de all_question_types
                global_row = all_data[all_data['question_type'] == 'GLOBAL']
                
                # Ordenar otros tipos según all_question_types
                other_rows = all_data[all_data['question_type'] != 'GLOBAL'].copy()
                other_rows['sort_order'] = other_rows['question_type'].apply(
                    lambda x: all_question_types.index(x) if x in all_question_types else len(all_question_types)
                )
                other_rows = other_rows.sort_values('sort_order').drop('sort_order', axis=1)
                
                plot_data = pd.concat([global_row, other_rows], ignore_index=True)
            else:
                # Solo datos globales
                plot_data = pd.DataFrame({
                    'question_type': ['GLOBAL'], 
                    metric: [overall_metrics[metric]]
                })
            
            # Crear colores usando el mapeo consistente
            colors = []
            for qtype in plot_data['question_type']:
                if qtype == 'GLOBAL':
                    colors.append('#2E86AB')
                else:
                    colors.append(type_colors.get(qtype, '#CCCCCC'))
            
            n_bars = len(plot_data)
            
            # Crear gráfica de barras
            bars = ax.bar(range(n_bars), plot_data[metric], color=colors, alpha=0.8)
            
            # Destacar barra global con borde negro
            if n_bars > 0:
                bars[0].set_edgecolor('black')
                bars[0].set_linewidth(2)
            
            # Configurar ejes
            ax.set_xticks(range(n_bars))
            ax.set_xticklabels(plot_data['question_type'], rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Valor', fontsize=9)
            ax.set_title(f'{metric}', fontsize=10, fontweight='bold')
            
            # Ajustar límite Y
            max_val = plot_data[metric].max()
            if max_val > 0:
                ax.set_ylim(0, max_val * 1.15)
            
            # Añadir valores sobre las barras
            for bar, value in zip(bars, plot_data[metric]):
                height = bar.get_height()
                if pd.notna(value):
                    label = f'{value:.3f}' if value < 10 else f'{value:.0f}'
                    ax.text(
                        bar.get_x() + bar.get_width() / 2., 
                        height,
                        label,
                        ha='center', 
                        va='bottom', 
                        fontsize=7
                    )
        
        # Ocultar subplots vacíos
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        # Título general
        plt.suptitle(f'{eval_name}', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Guardar
        safe_name = eval_name.replace(':', '_').replace('/', '_')
        output_file = output_dir / f'metrics_{safe_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Guardado: {output_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualiza resultados de experimentos de sistemas de memoria'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Ruta al archivo JSON de resultados'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='Directorio de salida para las gráficas (por defecto: junto al archivo de entrada)'
    )
    
    args = parser.parse_args()
    
    # Cargar resultados
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: El archivo {input_path} no existe")
        return
    
    print(f"Cargando resultados desde: {input_path}")
    results = load_results(input_path)
    
    # Determinar directorio de salida
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / f"{input_path.stem}_visualizations"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Guardando visualizaciones en: {output_dir}\n")
    
    # Generar visualizaciones
    print("Generando gráficas por evaluador...")
    plot_evaluator_metrics(results, output_dir)
    
    print(f"\n✓ Visualización completada.")


if __name__ == '__main__':
    main()
