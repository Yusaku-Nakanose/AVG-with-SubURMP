import os
import re
from collections import defaultdict

# One-hot encoding for instruments
INSTRUMENT_LABELS = {
    'bassoon': ('bn', '1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0'),
    'cello': ('vc', '0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0'),
    'clarinet': ('cl', '0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0'),
    'doublebass': ('db', '0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0'),
    'flute': ('fl', '0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0'),
    'horn': ('hn', '0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0'),
    'oboe': ('ob', '0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0'),
    'sax': ('sax', '0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0'),
    'trombone': ('tbn', '0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0'),
    'trumpet': ('tpt', '0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0'),
    'tuba': ('tba', '0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0'),
    'viola': ('va', '0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0'),
    'violin': ('vn', '0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1')
}

def get_piece_number(filename):
    """Extract piece number from filename"""
    pattern = r'[^_]+_[^_]+_(\d+)_\d+\.'
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1))
    return None

def get_frame_number(filename):
    """Extract frame number from filename"""
    pattern = r'[^_]+_[^_]+_\d+_(\d+)\.'
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1))
    return None

def group_files_by_piece(files):
    """Group files by piece number"""
    groups = defaultdict(list)
    for file in files:
        piece_num = get_piece_number(os.path.basename(file))
        if piece_num is not None:
            groups[piece_num].append(file)
    return {k: sorted(v, key=lambda x: get_frame_number(os.path.basename(x))) 
            for k, v in sorted(groups.items())}

def get_train_test_split(files, fold):
    """Get train/test split based on fold number"""
    splits = {
        1: (0.8, 1.0),    # fold_01: test on last 20%
        2: (0.6, 0.8),    # fold_02: test on 60-80%
        3: (0.4, 0.6),    # fold_03: test on 40-60%
        4: (0.2, 0.4),    # fold_04: test on 20-40%
        5: (0.0, 0.2)     # fold_05: test on first 20%
    }
    test_start, test_end = splits[fold]
    total_files = len(files)
    test_start_idx = round(test_start * total_files)
    test_end_idx = round(test_end * total_files)
    
    train_files = files[:test_start_idx] + files[test_end_idx:]
    test_files = files[test_start_idx:test_end_idx]
    
    return train_files, test_files

def get_consecutive_frames(sorted_frames, start_idx):
    """Get 4 consecutive frames starting from start_idx"""
    if start_idx + 4 > len(sorted_frames):
        return None
    return sorted_frames[start_idx:start_idx + 4]

def prepare_data_for_fold(dataset_root, output_root, fold, dataset_type='individual', log_file="prepare_data_3d_log.txt"):
    """Prepare 3D data for a specific fold"""
    with open(log_file, "a") as log:
        fold_dir = os.path.join(output_root, dataset_type, f'fold_{fold:02d}')
        os.makedirs(fold_dir, exist_ok=True)

        for instrument in INSTRUMENT_LABELS.keys():
            log.write(f"\nProcessing {instrument}...\n")
            
            # Get paths
            img_dir = os.path.join(dataset_root, 'img', instrument)
            spec_dir = os.path.join(dataset_root, 'spec', instrument)
            
            if not (os.path.exists(img_dir) and os.path.exists(spec_dir)):
                log.write(f"Warning: Paths not found for {instrument}\n")
                continue
            
            # Get all files and group by piece
            img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
            spec_files = [os.path.join(spec_dir, f) for f in os.listdir(spec_dir) if f.endswith('.png')]
            
            img_by_piece = group_files_by_piece(img_files)
            spec_by_piece = group_files_by_piece(spec_files)
            
            train_pairs = []
            test_pairs = []
            
            # Process each piece separately
            for piece_num in sorted(img_by_piece.keys()):
                piece_img_files = img_by_piece[piece_num]
                piece_spec_files = spec_by_piece.get(piece_num, [])
                
                # Check for expected difference in counts
                if len(piece_img_files) != len(piece_spec_files) + 4:
                    log.write(f"Error: Unexpected file count difference for piece {piece_num}\n")
                    continue
                
                # Split this piece's files according to fold
                train_spec, test_spec = get_train_test_split(piece_spec_files, fold)
                
                # Process train data
                for i, spec in enumerate(train_spec):
                    start_idx = i
                    frames = get_consecutive_frames(piece_img_files, start_idx)
                    if frames:
                        train_pairs.append((spec, frames))
                
                # Process test data
                for i, spec in enumerate(test_spec):
                    start_idx = i + len(train_spec)  # Adjust index based on train split
                    frames = get_consecutive_frames(piece_img_files, start_idx)
                    if frames:
                        test_pairs.append((spec, frames))
            
            # Create output files
            instrument_dir = os.path.join(fold_dir, instrument)
            os.makedirs(instrument_dir, exist_ok=True)
            
            short_name, label = INSTRUMENT_LABELS[instrument]
            
            # Write train file
            train_file = os.path.join(instrument_dir, f'train_{short_name}_3d.txt')
            with open(train_file, 'w') as f:
                for spec_file, img_files in train_pairs:
                    f.write(f"{spec_file}||{label}||{img_files[0]}||{img_files[1]}||{img_files[2]}||{img_files[3]}\n")
            
            # Write test file
            test_file = os.path.join(instrument_dir, f'test_{short_name}_3d.txt')
            with open(test_file, 'w') as f:
                for spec_file, img_files in test_pairs:
                    f.write(f"{spec_file}||{label}||{img_files[0]}||{img_files[1]}||{img_files[2]}||{img_files[3]}\n")
            
            log.write(f"Created files for {instrument}:\n")
            log.write(f"  Train file: {train_file}\n")
            log.write(f"  Test file: {test_file}\n")
            log.write(f"  Contents: {len(train_pairs)} train, {len(test_pairs)} test pairs\n")

def main():
    """Process both individual and ensemble datasets"""
    output_root = "/dlbox/avg-with-urmp/tseandtsl/data/data_links"
    log_file = os.path.join(output_root, "links_metadata_3d.txt")

    # Clear log file if exists
    if os.path.exists(log_file):
        os.remove(log_file)

    # Ensure output root directory exists
    os.makedirs(output_root, exist_ok=True)
    
    with open(log_file, "a") as log:
        log.write(f"Output directory: {output_root}\n")
    
    for dataset_type in ['individual', 'ensemble']:
        source_root = f"/dlbox/URMP_{dataset_type}"
        with open(log_file, "a") as log:
            log.write(f"\nProcessing {dataset_type} dataset from: {source_root}\n")
        for fold in range(1, 6):
            with open(log_file, "a") as log:
                log.write(f"\nCreating fold {fold}\n")
            prepare_data_for_fold(source_root, output_root, fold, dataset_type, log_file=log_file)

if __name__ == "__main__":
    main()