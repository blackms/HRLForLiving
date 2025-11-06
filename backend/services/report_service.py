"""Report generation service for creating PDF and HTML reports"""
import os
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json

from backend.utils.file_manager import (
    read_json_results,
    read_yaml_config,
    ensure_directories,
    BASE_DIR
)


# Report directories
REPORTS_DIR = BASE_DIR / "reports"
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


class ReportService:
    """Service for generating PDF and HTML reports from simulation results"""
    
    def __init__(self):
        """Initialize the report service"""
        ensure_directories()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    
    async def generate_report(
        self,
        simulation_id: str,
        report_type: str = "html",
        include_sections: Optional[List[str]] = None,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a report from simulation results
        
        Args:
            simulation_id: ID of the simulation results
            report_type: Type of report ('pdf' or 'html')
            include_sections: List of sections to include (None = all)
            title: Custom report title
            
        Returns:
            dict: Report metadata with file path and ID
            
        Raises:
            FileNotFoundError: If simulation results not found
            ValueError: If report type is invalid
        """
        # Validate report type
        if report_type not in ['pdf', 'html']:
            raise ValueError(f"Invalid report type: {report_type}. Must be 'pdf' or 'html'")
        
        # Load simulation results
        try:
            simulation_data = read_json_results(simulation_id, 'simulations')
        except FileNotFoundError:
            raise FileNotFoundError(f"Simulation results not found: {simulation_id}")
        
        # Load scenario configuration
        scenario_name = simulation_data.get('scenario_name', 'unknown')
        try:
            scenario_config = read_yaml_config(scenario_name, scenarios=True)
        except FileNotFoundError:
            scenario_config = None
        
        # Aggregate report data
        report_data = self._aggregate_report_data(
            simulation_data=simulation_data,
            scenario_config=scenario_config,
            title=title
        )
        
        # Determine sections to include
        all_sections = ['summary', 'scenario', 'training', 'results', 'charts', 'strategy']
        sections = include_sections if include_sections else all_sections
        report_data['sections'] = sections
        
        # Generate report based on type
        report_id = f"report_{simulation_id}_{int(datetime.now().timestamp())}"
        
        if report_type == 'html':
            report_path = self._generate_html_report(report_id, report_data)
        else:  # pdf
            report_path = self._generate_pdf_report(report_id, report_data)
        
        # Save report metadata
        metadata = {
            'report_id': report_id,
            'simulation_id': simulation_id,
            'report_type': report_type,
            'title': report_data['title'],
            'generated_at': datetime.now().isoformat(),
            'file_path': str(report_path.relative_to(BASE_DIR)),
            'file_size_kb': round(report_path.stat().st_size / 1024, 2),
            'sections': sections
        }
        
        metadata_path = REPORTS_DIR / f"{report_id}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def _aggregate_report_data(
        self,
        simulation_data: Dict[str, Any],
        scenario_config: Optional[Dict[str, Any]],
        title: Optional[str]
    ) -> Dict[str, Any]:
        """
        Aggregate all data needed for report generation
        
        Args:
            simulation_data: Simulation results
            scenario_config: Scenario configuration
            title: Custom title
            
        Returns:
            dict: Aggregated report data
        """
        # Extract key information
        scenario_name = simulation_data.get('scenario_name', 'Unknown')
        model_name = simulation_data.get('model_name', 'Unknown')
        
        # Default title
        if not title:
            title = f"Financial Simulation Report: {scenario_name}"
        
        # Build report data structure
        report_data = {
            'title': title,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'simulation_id': simulation_data.get('simulation_id', 'unknown'),
            'scenario_name': scenario_name,
            'model_name': model_name,
            'num_episodes': simulation_data.get('num_episodes', 0),
            'timestamp': simulation_data.get('timestamp', ''),
            
            # Summary statistics
            'summary': {
                'duration_mean': simulation_data.get('duration_mean', 0),
                'duration_std': simulation_data.get('duration_std', 0),
                'total_wealth_mean': simulation_data.get('total_wealth_mean', 0),
                'total_wealth_std': simulation_data.get('total_wealth_std', 0),
                'final_cash_mean': simulation_data.get('final_cash_mean', 0),
                'final_cash_std': simulation_data.get('final_cash_std', 0),
                'final_invested_mean': simulation_data.get('final_invested_mean', 0),
                'final_invested_std': simulation_data.get('final_invested_std', 0),
                'final_portfolio_mean': simulation_data.get('final_portfolio_mean', 0),
                'final_portfolio_std': simulation_data.get('final_portfolio_std', 0),
                'investment_gains_mean': simulation_data.get('investment_gains_mean', 0),
                'investment_gains_std': simulation_data.get('investment_gains_std', 0),
            },
            
            # Strategy learned
            'strategy': {
                'avg_invest_pct': simulation_data.get('avg_invest_pct', 0) * 100,
                'avg_save_pct': simulation_data.get('avg_save_pct', 0) * 100,
                'avg_consume_pct': simulation_data.get('avg_consume_pct', 0) * 100,
            },
            
            # Episode data for charts
            'episodes': simulation_data.get('episodes', []),
        }
        
        # Add scenario configuration if available
        if scenario_config:
            env_config = scenario_config.get('environment', {})
            report_data['scenario'] = {
                'description': scenario_config.get('description', ''),
                'income': env_config.get('income', 0),
                'fixed_expenses': env_config.get('fixed_expenses', 0),
                'variable_expense_mean': env_config.get('variable_expense_mean', 0),
                'variable_expense_std': env_config.get('variable_expense_std', 0),
                'inflation': env_config.get('inflation', 0) * 100,
                'safety_threshold': env_config.get('safety_threshold', 0),
                'initial_cash': env_config.get('initial_cash', 0),
                'risk_tolerance': env_config.get('risk_tolerance', 0) * 100,
                'investment_return_mean': env_config.get('investment_return_mean', 0) * 100,
                'investment_return_std': env_config.get('investment_return_std', 0) * 100,
                'investment_return_type': env_config.get('investment_return_type', 'stochastic'),
            }
            
            training_config = scenario_config.get('training', {})
            report_data['training'] = {
                'num_episodes': training_config.get('num_episodes', 0),
                'high_period': training_config.get('high_period', 6),
                'gamma_low': training_config.get('gamma_low', 0.95),
                'gamma_high': training_config.get('gamma_high', 0.99),
            }
        
        return report_data
    
    def _generate_html_report(self, report_id: str, report_data: Dict[str, Any]) -> Path:
        """
        Generate HTML report
        
        Args:
            report_id: Unique report identifier
            report_data: Aggregated report data
            
        Returns:
            Path to generated HTML file
        """
        # Build HTML content
        html_content = self._build_html_content(report_data)
        
        # Save HTML file
        report_path = REPORTS_DIR / f"{report_id}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def _build_html_content(self, report_data: Dict[str, Any]) -> str:
        """
        Build HTML content for the report
        
        Args:
            report_data: Aggregated report data
            
        Returns:
            HTML string
        """
        sections = report_data.get('sections', [])
        
        # Start HTML document
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_data['title']}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .metadata p {{
            margin: 5px 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-unit {{
            font-size: 0.8em;
            color: #95a5a6;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background-color: #34495e;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .strategy-bar {{
            display: flex;
            align-items: center;
            margin: 10px 0;
        }}
        .strategy-label {{
            width: 120px;
            font-weight: 600;
        }}
        .strategy-progress {{
            flex: 1;
            height: 30px;
            background-color: #ecf0f1;
            border-radius: 5px;
            overflow: hidden;
            margin: 0 15px;
        }}
        .strategy-fill {{
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .invest-fill {{ background-color: #3498db; }}
        .save-fill {{ background-color: #2ecc71; }}
        .consume-fill {{ background-color: #e74c3c; }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            text-align: center;
            color: #95a5a6;
            font-size: 0.9em;
        }}
        @media print {{
            body {{
                background-color: white;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{report_data['title']}</h1>
        
        <div class="metadata">
            <p><strong>Generated:</strong> {report_data['generated_at']}</p>
            <p><strong>Simulation ID:</strong> {report_data['simulation_id']}</p>
            <p><strong>Scenario:</strong> {report_data['scenario_name']}</p>
            <p><strong>Model:</strong> {report_data['model_name']}</p>
            <p><strong>Episodes:</strong> {report_data['num_episodes']}</p>
        </div>
"""
        
        # Summary section
        if 'summary' in sections:
            html += self._build_summary_section(report_data)
        
        # Scenario section
        if 'scenario' in sections and 'scenario' in report_data:
            html += self._build_scenario_section(report_data)
        
        # Training section
        if 'training' in sections and 'training' in report_data:
            html += self._build_training_section(report_data)
        
        # Results section
        if 'results' in sections:
            html += self._build_results_section(report_data)
        
        # Strategy section
        if 'strategy' in sections:
            html += self._build_strategy_section(report_data)
        
        # Charts section
        if 'charts' in sections:
            html += self._build_charts_section(report_data)
        
        # Footer
        html += f"""
        <div class="footer">
            <p>HRL Finance System - Hierarchical Reinforcement Learning for Personal Finance Optimization</p>
            <p>Report generated on {report_data['generated_at']}</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _build_summary_section(self, report_data: Dict[str, Any]) -> str:
        """Build summary statistics section"""
        summary = report_data['summary']
        
        return f"""
        <h2>Summary Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Average Duration</div>
                <div class="stat-value">{summary['duration_mean']:.1f} <span class="stat-unit">months</span></div>
                <div class="stat-unit">± {summary['duration_std']:.1f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Wealth</div>
                <div class="stat-value">{summary['total_wealth_mean']:,.0f} <span class="stat-unit">EUR</span></div>
                <div class="stat-unit">± {summary['total_wealth_std']:,.0f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Final Cash</div>
                <div class="stat-value">{summary['final_cash_mean']:,.0f} <span class="stat-unit">EUR</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Investment Gains</div>
                <div class="stat-value">{summary['investment_gains_mean']:,.0f} <span class="stat-unit">EUR</span></div>
                <div class="stat-unit">± {summary['investment_gains_std']:,.0f}</div>
            </div>
        </div>
"""
    
    def _build_scenario_section(self, report_data: Dict[str, Any]) -> str:
        """Build scenario configuration section"""
        scenario = report_data['scenario']
        
        return f"""
        <h2>Scenario Configuration</h2>
        <p><em>{scenario.get('description', 'No description provided')}</em></p>
        
        <h3>Financial Parameters</h3>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Monthly Income</td>
                <td>{scenario['income']:,.2f} EUR</td>
            </tr>
            <tr>
                <td>Fixed Expenses</td>
                <td>{scenario['fixed_expenses']:,.2f} EUR</td>
            </tr>
            <tr>
                <td>Variable Expenses (Mean)</td>
                <td>{scenario['variable_expense_mean']:,.2f} EUR</td>
            </tr>
            <tr>
                <td>Variable Expenses (Std Dev)</td>
                <td>{scenario['variable_expense_std']:,.2f} EUR</td>
            </tr>
            <tr>
                <td>Inflation Rate</td>
                <td>{scenario['inflation']:.2f}% per month</td>
            </tr>
            <tr>
                <td>Safety Threshold</td>
                <td>{scenario['safety_threshold']:,.2f} EUR</td>
            </tr>
            <tr>
                <td>Initial Cash</td>
                <td>{scenario['initial_cash']:,.2f} EUR</td>
            </tr>
            <tr>
                <td>Risk Tolerance</td>
                <td>{scenario['risk_tolerance']:.1f}%</td>
            </tr>
        </table>
        
        <h3>Investment Parameters</h3>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Return Type</td>
                <td>{scenario['investment_return_type'].capitalize()}</td>
            </tr>
            <tr>
                <td>Expected Return (Mean)</td>
                <td>{scenario['investment_return_mean']:.2f}% per month</td>
            </tr>
            <tr>
                <td>Return Volatility (Std Dev)</td>
                <td>{scenario['investment_return_std']:.2f}% per month</td>
            </tr>
        </table>
"""
    
    def _build_training_section(self, report_data: Dict[str, Any]) -> str:
        """Build training configuration section"""
        training = report_data['training']
        
        return f"""
        <h2>Training Configuration</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Training Episodes</td>
                <td>{training['num_episodes']:,}</td>
            </tr>
            <tr>
                <td>High-Level Planning Period</td>
                <td>{training['high_period']} months</td>
            </tr>
            <tr>
                <td>Low-Level Discount Factor (γ)</td>
                <td>{training['gamma_low']:.3f}</td>
            </tr>
            <tr>
                <td>High-Level Discount Factor (γ)</td>
                <td>{training['gamma_high']:.3f}</td>
            </tr>
        </table>
"""
    
    def _build_results_section(self, report_data: Dict[str, Any]) -> str:
        """Build detailed results section"""
        summary = report_data['summary']
        
        return f"""
        <h2>Detailed Results</h2>
        
        <h3>Wealth Breakdown</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Mean</th>
                <th>Std Dev</th>
            </tr>
            <tr>
                <td>Final Cash Balance</td>
                <td>{summary['final_cash_mean']:,.2f} EUR</td>
                <td>-</td>
            </tr>
            <tr>
                <td>Total Invested</td>
                <td>{summary['final_invested_mean']:,.2f} EUR</td>
                <td>-</td>
            </tr>
            <tr>
                <td>Portfolio Value</td>
                <td>{summary['final_portfolio_mean']:,.2f} EUR</td>
                <td>{summary['final_portfolio_std']:,.2f} EUR</td>
            </tr>
            <tr>
                <td><strong>Total Wealth</strong></td>
                <td><strong>{summary['total_wealth_mean']:,.2f} EUR</strong></td>
                <td><strong>{summary['total_wealth_std']:,.2f} EUR</strong></td>
            </tr>
        </table>
"""
    
    def _build_strategy_section(self, report_data: Dict[str, Any]) -> str:
        """Build strategy learned section"""
        strategy = report_data['strategy']
        
        return f"""
        <h2>Strategy Learned</h2>
        <p>Average allocation of available funds across all episodes:</p>
        
        <div class="strategy-bar">
            <div class="strategy-label">Invest</div>
            <div class="strategy-progress">
                <div class="strategy-fill invest-fill" style="width: {strategy['avg_invest_pct']:.1f}%">
                    {strategy['avg_invest_pct']:.1f}%
                </div>
            </div>
        </div>
        
        <div class="strategy-bar">
            <div class="strategy-label">Save</div>
            <div class="strategy-progress">
                <div class="strategy-fill save-fill" style="width: {strategy['avg_save_pct']:.1f}%">
                    {strategy['avg_save_pct']:.1f}%
                </div>
            </div>
        </div>
        
        <div class="strategy-bar">
            <div class="strategy-label">Consume</div>
            <div class="strategy-progress">
                <div class="strategy-fill consume-fill" style="width: {strategy['avg_consume_pct']:.1f}%">
                    {strategy['avg_consume_pct']:.1f}%
                </div>
            </div>
        </div>
"""
    
    def _build_charts_section(self, report_data: Dict[str, Any]) -> str:
        """Build charts section with episode data"""
        episodes = report_data.get('episodes', [])
        
        if not episodes:
            return "<h2>Charts</h2><p>No episode data available for visualization.</p>"
        
        # Get first episode for visualization
        first_episode = episodes[0]
        
        return f"""
        <h2>Episode Visualization</h2>
        <p><em>Note: Interactive charts are best viewed in the web application. This report shows data from the first episode.</em></p>
        
        <h3>Episode Summary</h3>
        <table>
            <tr>
                <th>Episode</th>
                <th>Duration (months)</th>
                <th>Final Cash</th>
                <th>Final Portfolio</th>
                <th>Total Wealth</th>
            </tr>
"""
        
        # Add rows for each episode
        for ep in episodes[:10]:  # Show first 10 episodes
            html_row = f"""
            <tr>
                <td>Episode {ep['episode_id'] + 1}</td>
                <td>{ep['duration']}</td>
                <td>{ep['final_cash']:,.2f} EUR</td>
                <td>{ep['final_portfolio_value']:,.2f} EUR</td>
                <td>{ep['total_wealth']:,.2f} EUR</td>
            </tr>
"""
        
        if len(episodes) > 10:
            html_row += f"""
            <tr>
                <td colspan="5" style="text-align: center; font-style: italic;">
                    ... and {len(episodes) - 10} more episodes
                </td>
            </tr>
"""
        
        html_row += """
        </table>
"""
        
        return html_row
    
    def _generate_pdf_report(self, report_id: str, report_data: Dict[str, Any]) -> Path:
        """
        Generate PDF report using WeasyPrint
        
        Args:
            report_id: Unique report identifier
            report_data: Aggregated report data
            
        Returns:
            Path to generated PDF file
        """
        try:
            from weasyprint import HTML
        except ImportError:
            # Fallback: generate HTML and inform user
            html_path = self._generate_html_report(report_id, report_data)
            raise ImportError(
                f"WeasyPrint not installed. HTML report generated at {html_path}. "
                "Install WeasyPrint with: pip install weasyprint"
            )
        
        # Generate HTML content
        html_content = self._build_html_content(report_data)
        
        # Convert to PDF
        report_path = REPORTS_DIR / f"{report_id}.pdf"
        HTML(string=html_content).write_pdf(report_path)
        
        return report_path
    
    def get_report(self, report_id: str) -> Dict[str, Any]:
        """
        Get report metadata and file path
        
        Args:
            report_id: Unique report identifier
            
        Returns:
            dict: Report metadata
            
        Raises:
            FileNotFoundError: If report not found
        """
        metadata_path = REPORTS_DIR / f"{report_id}_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Report not found: {report_id}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return metadata
    
    def list_reports(self) -> List[Dict[str, Any]]:
        """
        List all generated reports
        
        Returns:
            list: List of report metadata
        """
        if not REPORTS_DIR.exists():
            return []
        
        reports = []
        for metadata_file in REPORTS_DIR.glob('*_metadata.json'):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                reports.append(metadata)
            except Exception as e:
                print(f"Warning: Could not read report metadata {metadata_file}: {e}")
                continue
        
        # Sort by generation time (newest first)
        reports.sort(key=lambda x: x.get('generated_at', ''), reverse=True)
        
        return reports
    
    def get_report_file_path(self, report_id: str) -> Path:
        """
        Get the file path for a report
        
        Args:
            report_id: Unique report identifier
            
        Returns:
            Path to report file
            
        Raises:
            FileNotFoundError: If report file not found
        """
        # Try HTML first
        html_path = REPORTS_DIR / f"{report_id}.html"
        if html_path.exists():
            return html_path
        
        # Try PDF
        pdf_path = REPORTS_DIR / f"{report_id}.pdf"
        if pdf_path.exists():
            return pdf_path
        
        raise FileNotFoundError(f"Report file not found: {report_id}")


# Global report service instance
report_service = ReportService()
