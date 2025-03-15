import csv
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime
import numpy as np
import torch

class CSVLogger:
    """A logger that writes metrics to CSV files, similar to TensorBoard's SummaryWriter.
    
    Args:
        log_dir (str): Directory where the CSV files will be saved
        flush_every (int, optional): How often to flush the CSV file to disk. Defaults to 1.
    """
    def __init__(self, log_dir: str, flush_every: int = 1):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self.writers: Dict[str, csv.DictWriter] = {}
        self.files: Dict[str, Any] = {}
        self.write_counts: Dict[str, int] = {}
        self.fieldnames: Dict[str, list] = {}
        
    def _convert_value(self, value: Any) -> Union[float, int]:
        """Convert a value to either float or int, preserving integer types."""
        if isinstance(value, (int, np.integer)):
            return int(value)
        try:
            if isinstance(value, (torch.Tensor, np.ndarray)):
                value = value.item()
            if isinstance(value, (int, np.integer)):
                return int(value)
            return float(value)
        except:
            return float(np.array(value))
        
    def add_scalar(self, tag: str, scalar_value: Union[float, Any], global_step: Optional[int] = None, metadata: Optional[Dict] = None):
        """Add scalar data to the CSV file.
        
        Args:
            tag (str): Data identifier
            scalar_value (float or tensor-like): Value to save
            global_step (int, optional): Global step value to record
            metadata (dict, optional): Additional metadata to include in the CSV
        """
        # Convert the scalar value while preserving integer type if applicable
        scalar_value = self._convert_value(scalar_value)
            
        # Get the appropriate writer or create a new one
        if tag not in self.writers:
            filepath = self.log_dir / tag
            # Create parent directories if they don't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)
            f = open(filepath.with_suffix('.csv'), 'w', newline='')
            self.files[tag] = f
            
            # Define fieldnames including metadata keys if provided
            self.fieldnames[tag] = ['timestamp', 'step', 'value']
            if metadata is not None:
                self.fieldnames[tag].extend(metadata.keys())
            
            writer = csv.DictWriter(f, fieldnames=self.fieldnames[tag])
            writer.writeheader()
            
            self.writers[tag] = writer
            self.write_counts[tag] = 0
            
        # Prepare row data
        row_data = {
            'timestamp': datetime.now().isoformat(),
            'step': int(global_step) if global_step is not None else self.write_counts[tag],
            'value': scalar_value
        }
        
        # Add metadata if provided, converting values appropriately
        if metadata is not None:
            metadata_converted = {
                k: self._convert_value(v) if isinstance(v, (int, float, np.number, torch.Tensor)) else v
                for k, v in metadata.items()
            }
            row_data.update(metadata_converted)
            
            # If new metadata keys are present, create a new writer with updated fieldnames
            new_keys = set(metadata.keys()) - set(self.fieldnames[tag])
            if new_keys:
                self.fieldnames[tag].extend(new_keys)
                old_file = self.files[tag]
                old_file.close()
                
                # Create new file with updated headers
                filepath = self.log_dir / tag
                f = open(filepath.with_suffix('.csv'), 'w', newline='')
                self.files[tag] = f
                writer = csv.DictWriter(f, fieldnames=self.fieldnames[tag])
                writer.writeheader()
                self.writers[tag] = writer
        
        # Write the data
        self.writers[tag].writerow(row_data)
        
        self.write_counts[tag] += 1
        
        # Flush if needed
        if self.write_counts[tag] % self.flush_every == 0:
            self.files[tag].flush()
            
    def close(self):
        """Close all open file handles."""
        for f in self.files.values():
            f.close()
        self.writers.clear()
        self.files.clear()
        self.write_counts.clear()
        self.fieldnames.clear()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == '__main__':
    # Example usage
    with CSVLogger("logs/example") as logger:
        # Log some test metrics with metadata
        for i in range(10):
            metadata = {"model_id": i % 2, "batch_size": 32}
            logger.add_scalar("loss", 1.0 / (i + 1), global_step=i, metadata=metadata)
            logger.add_scalar("accuracy", i / 10.0, global_step=i, metadata=metadata) 