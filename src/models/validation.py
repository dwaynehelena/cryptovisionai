#!/usr/bin/env python3
"""
Model validation utilities for CryptoVisionAI models
Includes walk-forward cross-validation and backtesting utilities
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Union, Any, Optional, Callable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json

# Configure logging
logger = logging.getLogger("model_validation")

class WalkForwardValidator:
    """
    Walk-forward cross-validation for time series data
    
    This validation strategy respects time order and avoids look-ahead bias by:
    1. Training on a sliding window of data
    2. Validating on data immediately following the training window
    3. Moving the window forward and repeating
    """
    
    def __init__(
        self, 
        n_splits: int = 5, 
        train_size: float = 0.7,
        gap: int = 0,
        step_size: Optional[int] = None
    ):
        """
        Initialize WalkForwardValidator
        
        Args:
            n_splits: Number of validation splits to perform
            train_size: Proportion of data to use for training in each split
            gap: Number of samples to exclude between train and test sets
            step_size: Number of samples to shift forward between splits
                     (if None, will use test_size)
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.gap = gap
        self.step_size = step_size
        
        logger.info(f"Initialized WalkForwardValidator with {n_splits} splits, train_size={train_size}")
            
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward train/test indices
        
        Args:
            X: Feature matrix (time-ordered)
            y: Target vector (optional)
            
        Returns:
            List of (train_indices, test_indices) tuples for each split
        """
        n_samples = len(X)
        
        # Calculate sizes
        train_size = int(n_samples * self.train_size)
        test_size = int((n_samples - train_size - self.gap) / self.n_splits)
        
        if test_size <= 0:
            raise ValueError(
                f"Insufficient data for {self.n_splits} splits. "
                f"Have {n_samples} samples, need at least "
                f"{int(self.n_splits / (1 - self.train_size))} samples."
            )
        
        # Calculate step size if not provided
        step = self.step_size if self.step_size is not None else test_size
        
        # Generate splits
        splits = []
        
        for i in range(self.n_splits):
            # Calculate start indices for this split
            train_start = i * step
            train_end = train_start + train_size
            test_start = train_end + self.gap
            test_end = test_start + test_size
            
            # Make sure we don't go beyond the data
            if test_end > n_samples:
                if i == 0:  # If even first split doesn't fit
                    raise ValueError(
                        f"Dataset too small for validation with current parameters. "
                        f"Have {n_samples} samples, first split needs {test_end}."
                    )
                break
            
            # Create indices
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
            
        logger.info(f"Created {len(splits)} walk-forward validation splits")
        return splits
    
    def validate(
        self, 
        X: pd.DataFrame,
        y: pd.Series, 
        model_initializer: Callable[[], Any],
        model_trainer: Callable[[Any, pd.DataFrame, pd.Series], Any],
        model_predictor: Callable[[Any, pd.DataFrame], np.ndarray],
        metrics: List[str] = ['accuracy', 'f1', 'roc_auc']
    ) -> Dict[str, Any]:
        """
        Perform walk-forward validation with provided model functions
        
        Args:
            X: Feature matrix (time-ordered)
            y: Target variable
            model_initializer: Function to initialize the model
            model_trainer: Function to train the model (takes model, X_train, y_train)
            model_predictor: Function to generate predictions (takes model, X_test)
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary with validation results for each fold and overall metrics
        """
        logger.info("Starting walk-forward validation")
        
        # Generate splits
        splits = self.split(X, y)
        
        # Store results for each fold
        fold_results = []
        all_predictions = []
        all_actual = []
        
        # Track computation time
        start_time = datetime.now()
        
        # Perform validation
        for fold, (train_indices, test_indices) in enumerate(splits):
            fold_start_time = datetime.now()
            
            # Get train/test data for this fold
            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_test = y.iloc[test_indices]
            
            # Initialize and train model
            logger.info(f"Fold {fold+1}/{len(splits)}: Training model on {len(X_train)} samples")
            model = model_initializer()
            model = model_trainer(model, X_train, y_train)
            
            # Generate predictions
            y_pred_proba = model_predictor(model, X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            fold_metric_values = {}
            for metric in metrics:
                if metric == 'accuracy':
                    value = accuracy_score(y_test, y_pred)
                elif metric == 'precision':
                    value = precision_score(y_test, y_pred, zero_division=0)
                elif metric == 'recall':
                    value = recall_score(y_test, y_pred, zero_division=0)
                elif metric == 'f1':
                    value = f1_score(y_test, y_pred, zero_division=0)
                elif metric == 'roc_auc':
                    # Handle case where all classes are the same
                    try:
                        value = roc_auc_score(y_test, y_pred_proba)
                    except:
                        value = 0.5
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                    
                fold_metric_values[metric] = value
                
            # Calculate fold duration
            fold_duration = (datetime.now() - fold_start_time).total_seconds()
            
            # Save fold results
            fold_results.append({
                'fold': fold + 1,
                'train_start': int(train_indices[0]),
                'train_end': int(train_indices[-1]),
                'test_start': int(test_indices[0]),
                'test_end': int(test_indices[-1]),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'metrics': fold_metric_values,
                'duration_seconds': fold_duration
            })
            
            logger.info(f"Fold {fold+1}/{len(splits)} complete: {fold_metric_values}")
            
            # Store predictions for overall metrics
            all_predictions.extend(y_pred_proba)
            all_actual.extend(y_test.values)
        
        # Calculate overall metrics
        overall_metrics = {}
        all_predictions = np.array(all_predictions)
        all_actual = np.array(all_actual)
        all_pred_classes = (all_predictions > 0.5).astype(int)
        
        for metric in metrics:
            if metric == 'accuracy':
                value = accuracy_score(all_actual, all_pred_classes)
            elif metric == 'precision':
                value = precision_score(all_actual, all_pred_classes, zero_division=0)
            elif metric == 'recall':
                value = recall_score(all_actual, all_pred_classes, zero_division=0)
            elif metric == 'f1':
                value = f1_score(all_actual, all_pred_classes, zero_division=0)
            elif metric == 'roc_auc':
                try:
                    value = roc_auc_score(all_actual, all_predictions)
                except:
                    value = 0.5
                    
            overall_metrics[metric] = value
        
        # Calculate average metrics across folds
        avg_fold_metrics = {}
        for metric in metrics:
            values = [fold['metrics'][metric] for fold in fold_results]
            avg_fold_metrics[metric] = sum(values) / len(values)
            avg_fold_metrics[f"{metric}_std"] = np.std(values)
        
        # Calculate total duration
        total_duration = (datetime.now() - start_time).total_seconds()
        
        # Compile results
        results = {
            'overall_metrics': overall_metrics,
            'average_fold_metrics': avg_fold_metrics,
            'fold_results': fold_results,
            'n_splits': len(splits),
            'train_size': self.train_size,
            'gap': self.gap,
            'total_samples': len(X),
            'total_duration_seconds': total_duration
        }
        
        logger.info(f"Walk-forward validation complete. Overall metrics: {overall_metrics}")
        return results
    
    def plot_results(
        self, 
        results: Dict[str, Any],
        metric: str = 'roc_auc',
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot validation results across folds
        
        Args:
            results: Results dictionary from validate method
            metric: Metric to plot
            save_path: Path to save plot (if None, will display)
        """
        plt.figure(figsize=(12, 6))
        
        # Extract values to plot
        folds = [r['fold'] for r in results['fold_results']]
        values = [r['metrics'][metric] for r in results['fold_results']]
        
        # Plot metrics across folds
        sns.lineplot(x=folds, y=values, marker='o', linewidth=2)
        
        # Add average line
        avg_value = results['average_fold_metrics'][metric]
        plt.axhline(y=avg_value, color='r', linestyle='--', 
                   label=f'Average: {avg_value:.4f}')
        
        # Add overall line 
        overall_value = results['overall_metrics'][metric]
        plt.axhline(y=overall_value, color='g', linestyle='--', 
                   label=f'Overall: {overall_value:.4f}')
        
        # Add labels and title
        plt.xlabel('Fold')
        plt.ylabel(metric.upper())
        plt.title(f'Walk-Forward Validation: {metric.upper()} Across Folds')
        plt.xticks(folds)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save or show
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_results(
        self, 
        results: Dict[str, Any],
        save_path: str,
        model_name: str = "model",
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save validation results to JSON file
        
        Args:
            results: Results dictionary from validate method
            save_path: Directory to save results
            model_name: Name of the model
            additional_info: Additional information to include
        """
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_validation_{timestamp}.json"
        file_path = os.path.join(save_path, filename)
        
        # Add additional information if provided
        output = results.copy()
        if additional_info:
            output.update(additional_info)
            
        # Add timestamp
        output['timestamp'] = timestamp
        output['model_name'] = model_name
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
            
        logger.info(f"Validation results saved to {file_path}")


class TimeSeriesBacktester:
    """
    Time series backtesting for trading strategy validation
    
    This backtester uses walk-forward validation and adds trading-specific metrics:
    - Profit/loss metrics
    - Drawdown analysis
    - Risk-adjusted return measures
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,  # 0.1% per trade
        position_sizing: str = 'equal',  # 'equal', 'kelly', 'percent'
        position_size: float = 0.1,  # 10% of capital per trade
        stop_loss: Optional[float] = None,  # e.g., 0.05 for 5% stop loss
        take_profit: Optional[float] = None  # e.g., 0.1 for 10% take profit
    ):
        """
        Initialize TimeSeriesBacktester
        
        Args:
            initial_capital: Initial capital for backtesting
            transaction_cost: Transaction cost as a fraction of trade value
            position_sizing: Position sizing strategy
            position_size: Size of position (meaning depends on strategy)
            stop_loss: Stop loss as a fraction of entry price (optional)
            take_profit: Take profit as a fraction of entry price (optional)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.position_sizing = position_sizing
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Validate parameters
        if position_sizing not in ['equal', 'kelly', 'percent']:
            raise ValueError(f"Invalid position_sizing: {position_sizing}")
        
        if position_size <= 0 or position_size > 1:
            raise ValueError(f"position_size must be between 0 and 1, got {position_size}")
            
        logger.info(f"Initialized TimeSeriesBacktester with initial_capital={initial_capital}, "
                   f"transaction_cost={transaction_cost}, position_sizing={position_sizing}")
    
    def backtest(
        self,
        price_data: pd.DataFrame,
        signals: np.ndarray,
        signal_probabilities: Optional[np.ndarray] = None,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, Any]:
        """
        Backtest a trading strategy using historical price data and signals
        
        Args:
            price_data: DataFrame with price data (must include 'close')
            signals: Binary signals array (1 for buy, 0 for hold/sell)
            signal_probabilities: Probabilities for each signal (optional)
            dates: DatetimeIndex for the signals (optional, falls back to price_data.index)
            
        Returns:
            Dictionary with backtesting results
        """
        logger.info("Starting backtesting")
        
        # Ensure we have close prices
        if 'close' not in price_data.columns:
            raise ValueError("price_data must contain 'close' column")
            
        # Use dates from price_data if not provided
        if dates is None:
            dates = price_data.index
            
        # Ensure all arrays have the same length
        n_signals = len(signals)
        if len(price_data) != n_signals:
            raise ValueError(f"price_data length ({len(price_data)}) "
                           f"does not match signals length ({n_signals})")
        
        # Initialize results tracking
        capital = self.initial_capital
        positions = 0
        trades = []
        equity_curve = [capital]
        returns = []
        
        # Track metrics
        n_trades = 0
        n_winning = 0
        n_losing = 0
        total_profit = 0.0
        total_loss = 0.0
        
        # Process each signal
        for i in range(1, n_signals):
            current_price = price_data['close'].iloc[i]
            prev_price = price_data['close'].iloc[i-1]
            signal = signals[i]
            date = dates[i]
            
            # Calculate price return
            price_return = current_price / prev_price - 1
            
            # Check if we have an open position
            if positions > 0:
                # Apply price change to current position
                position_value_change = positions * price_return
                capital += position_value_change
                
                # Check for stop loss or take profit
                if self._check_exit_conditions(current_price, entry_price, signal):
                    # Close position
                    exit_value = positions * current_price * (1 - self.transaction_cost)
                    trade_profit = exit_value - entry_value
                    
                    # Record trade
                    trade_end = {
                        'exit_date': date,
                        'exit_price': current_price,
                        'exit_value': exit_value,
                        'profit': trade_profit,
                        'profit_pct': trade_profit / entry_value
                    }
                    trades[-1].update(trade_end)
                    
                    # Update metrics
                    if trade_profit > 0:
                        n_winning += 1
                        total_profit += trade_profit
                    else:
                        n_losing += 1
                        total_loss += abs(trade_profit)
                    
                    # Reset position
                    capital = exit_value
                    positions = 0
                    
            # Check for new signal
            if signal == 1 and positions == 0:
                # Calculate position size
                size = self._calculate_position_size(capital, signal_probabilities[i] if signal_probabilities is not None else None)
                
                # Enter position
                entry_value = size * (1 - self.transaction_cost)
                positions = entry_value / current_price
                entry_price = current_price
                
                # Record trade
                trade = {
                    'entry_date': date,
                    'entry_price': current_price,
                    'entry_value': entry_value,
                    'size': size,
                    'probability': signal_probabilities[i] if signal_probabilities is not None else None
                }
                trades.append(trade)
                
                # Update metrics
                n_trades += 1
                
                # Deduct entry value from capital
                capital -= size
            
            # Record equity and returns
            equity_curve.append(capital + (positions * current_price if positions > 0 else 0))
            returns.append((equity_curve[-1] / equity_curve[-2]) - 1 if i > 0 else 0)
        
        # Close any remaining position at the end
        if positions > 0:
            final_price = price_data['close'].iloc[-1]
            exit_value = positions * final_price * (1 - self.transaction_cost)
            trade_profit = exit_value - entry_value
            
            # Record final trade
            trade_end = {
                'exit_date': dates[-1],
                'exit_price': final_price,
                'exit_value': exit_value,
                'profit': trade_profit,
                'profit_pct': trade_profit / entry_value
            }
            trades[-1].update(trade_end)
            
            # Update metrics
            if trade_profit > 0:
                n_winning += 1
                total_profit += trade_profit
            else:
                n_losing += 1
                total_loss += abs(trade_profit)
                
            # Add to capital
            capital = exit_value + capital
            
            # Update final equity
            equity_curve[-1] = capital
        
        # Convert to numpy array for calculations
        equity_curve = np.array(equity_curve)
        returns = np.array(returns)
        
        # Calculate performance metrics
        final_equity = equity_curve[-1]
        total_return = (final_equity / self.initial_capital) - 1
        
        # Calculate drawdown
        drawdown = self._calculate_drawdown(equity_curve)
        max_drawdown = drawdown.max()
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        
        # Win rate and profit factor
        win_rate = n_winning / n_trades if n_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Average profit per trade
        avg_profit = (total_profit - total_loss) / n_trades if n_trades > 0 else 0
        
        # Results dictionary
        results = {
            'equity_curve': equity_curve.tolist(),
            'returns': returns.tolist(),
            'drawdown': drawdown.tolist(),
            'trades': trades,
            'metrics': {
                'initial_capital': self.initial_capital,
                'final_equity': final_equity,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'annualized_return': self._calculate_annualized_return(total_return, len(returns)),
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'num_trades': n_trades,
                'num_winning': n_winning,
                'num_losing': n_losing,
                'avg_profit_per_trade': avg_profit,
                'avg_profit_pct_per_trade': avg_profit / self.initial_capital * 100 if n_trades > 0 else 0
            }
        }
        
        logger.info(f"Backtesting complete. Total return: {total_return * 100:.2f}%, "
                   f"Max drawdown: {max_drawdown * 100:.2f}%, Win rate: {win_rate * 100:.2f}%")
        
        return results
    
    def plot_equity_curve(
        self, 
        results: Dict[str, Any],
        benchmark: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot equity curve and drawdown
        
        Args:
            results: Results from backtest method
            benchmark: Benchmark returns (optional)
            save_path: Path to save plot (if None, will display)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Equity curve
        equity = np.array(results['equity_curve'])
        ax1.plot(equity, label='Strategy', linewidth=2, color='blue')
        
        # Add benchmark if provided
        if benchmark is not None:
            # Normalize benchmark to same starting capital
            norm_benchmark = benchmark / benchmark[0] * self.initial_capital
            ax1.plot(norm_benchmark, label='Benchmark', linewidth=1, color='gray', alpha=0.7)
        
        # Add buy/sell markers
        for trade in results['trades']:
            entry_idx = None
            exit_idx = None
            
            # Find indices based on dates
            for i, date in enumerate(trade['entry_date']):
                if date == trade['entry_date']:
                    entry_idx = i
                    break
            
            if 'exit_date' in trade:
                for i, date in enumerate(trade['exit_date']):
                    if date == trade['exit_date']:
                        exit_idx = i
                        break
            
            # Plot entry point
            if entry_idx is not None:
                ax1.plot(entry_idx, equity[entry_idx], 'g^', markersize=8)
            
            # Plot exit point
            if exit_idx is not None:
                ax1.plot(exit_idx, equity[exit_idx], 'rv', markersize=8)
        
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Drawdown plot
        drawdown = np.array(results['drawdown'])
        ax2.fill_between(range(len(drawdown)), 0, -drawdown, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Drawdown')
        ax2.grid(True, alpha=0.3)
        
        # Add metrics as text box
        metrics = results['metrics']
        metrics_text = (
            f"Total Return: {metrics['total_return_pct']:.2f}%\n"
            f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Win Rate: {metrics['win_rate'] * 100:.2f}%"
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.05, 0.05, metrics_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='bottom', bbox=props)
        
        # Adjust layout and display/save
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def _calculate_position_size(
        self,
        capital: float,
        probability: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on the chosen strategy
        
        Args:
            capital: Available capital
            probability: Signal probability (used for Kelly criterion)
            
        Returns:
            Position size in capital units
        """
        if self.position_sizing == 'equal':
            # Equal position sizing - fixed fraction of capital
            return capital * self.position_size
            
        elif self.position_sizing == 'percent':
            # Percent risk - risk a percentage of capital
            return capital * self.position_size
            
        elif self.position_sizing == 'kelly':
            # Kelly criterion - optimal position size based on win probability
            # and payoff ratio
            if probability is None:
                # Fall back to equal sizing if no probability is given
                return capital * self.position_size
            
            # Simplified Kelly: f = p - (1-p)/reward_ratio
            # Assuming 1:1 reward ratio
            reward_ratio = 1.0
            kelly_percent = probability - (1 - probability) / reward_ratio
            
            # Limit Kelly to position_size and ensure it's positive
            kelly_percent = max(0, min(kelly_percent, self.position_size))
            
            return capital * kelly_percent
    
    def _check_exit_conditions(
        self,
        current_price: float,
        entry_price: float,
        signal: int
    ) -> bool:
        """
        Check if exit conditions are met
        
        Args:
            current_price: Current price
            entry_price: Entry price
            signal: Current signal
            
        Returns:
            True if should exit position, False otherwise
        """
        # Exit on signal reversal (signal=0)
        if signal == 0:
            return True
        
        # Check stop loss
        if self.stop_loss and current_price <= entry_price * (1 - self.stop_loss):
            return True
        
        # Check take profit
        if self.take_profit and current_price >= entry_price * (1 + self.take_profit):
            return True
        
        return False
    
    def _calculate_drawdown(self, equity_curve: np.ndarray) -> np.ndarray:
        """
        Calculate drawdown series
        
        Args:
            equity_curve: Array of equity values
            
        Returns:
            Array of drawdown values
        """
        # Running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        return -drawdown  # Return as positive values
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
            
        # Assume daily returns, annualize by sqrt(252)
        excess_returns = returns - risk_free_rate / 252
        
        # Check for valid data
        if np.std(returns) == 0:
            return 0.0
            
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino ratio
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
            
        # Assume daily returns, annualize 
        excess_returns = returns - risk_free_rate / 252
        
        # Get negative returns only
        negative_returns = excess_returns[excess_returns < 0]
        
        # Check for valid data
        if len(negative_returns) == 0 or np.std(negative_returns) == 0:
            return 0.0 if np.mean(excess_returns) <= 0 else float('inf')
            
        sortino = np.mean(excess_returns) / np.std(negative_returns) * np.sqrt(252)
        return sortino
    
    def _calculate_annualized_return(self, total_return: float, num_periods: int) -> float:
        """
        Calculate annualized return
        
        Args:
            total_return: Total return
            num_periods: Number of periods
            
        Returns:
            Annualized return
        """
        # Assume daily periods
        if num_periods < 1:
            return 0.0
            
        # (1 + r)^(1/t) - 1
        years = num_periods / 252
        annualized = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        return annualized