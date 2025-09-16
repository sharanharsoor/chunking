# Large Files Test Directory

This directory is for testing streaming functionality with massive files (GB+ size).

## ⚠️ Important Note

**Large test files are NOT committed to git** to keep the repository lightweight. You need to create your own test files locally.

## 📁 Expected Test Files

The streaming tests expect the following large files:

### Required Files for Full Testing
- `repeated.txt` - Large text file with repeated content (5GB+ recommended)
- `random.txt` - Large file with random binary/text data (10GB+ recommended)
- `huge_dataset.csv` - Large CSV file (2GB+ recommended)
- `massive_log.log` - Large log file with structured entries (1GB+ recommended)

### Optional Files for Advanced Testing
- `giant_document.pdf` - Large PDF document (500MB+ recommended)
- `big_json.json` - Large JSON file with nested structures (1GB+ recommended)
- `enormous_code.py` - Large Python source file (100MB+ recommended)

## 🚀 Quick File Generation

You can create test files using these commands:

### Create Large Text File (5GB)
```bash
# Repeated content (like your current repeated.txt)
python3 -c "
content = 'HelloWorld\n'
with open('repeated.txt', 'w') as f:
    for _ in range(500_000_000):  # 5GB of 'HelloWorld\n'
        f.write(content)
print('✅ Created repeated.txt (~5GB)')
"
```

### Create Random Data File (10GB)
```bash
# Random binary data (like your current random.txt)
python3 -c "
import random
import string
with open('random.txt', 'wb') as f:
    for _ in range(10_000_000):  # 10GB in 1KB chunks
        chunk = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + '\n', k=1024))
        f.write(chunk.encode('utf-8', errors='ignore'))
print('✅ Created random.txt (~10GB)')
"
```

### Create Large CSV (2GB)
```bash
python3 -c "
import csv
import random
with open('huge_dataset.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'name', 'email', 'data', 'timestamp', 'value'])
    for i in range(50_000_000):  # 2GB+ of CSV data
        writer.writerow([
            i,
            f'user_{random.randint(1000,9999)}',
            f'user{i}@example.com',
            ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=20)),
            f'2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
            random.random() * 1000
        ])
        if i % 1_000_000 == 0:
            print(f'Progress: {i:,} rows written')
print('✅ Created huge_dataset.csv (~2GB)')
"
```

### Create Large Log File (1GB)
```bash
python3 -c "
import random
import datetime
log_levels = ['INFO', 'DEBUG', 'WARN', 'ERROR', 'TRACE']
components = ['auth', 'db', 'api', 'cache', 'worker', 'scheduler']
with open('massive_log.log', 'w') as f:
    for i in range(20_000_000):  # 1GB+ of log entries
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        level = random.choice(log_levels)
        component = random.choice(components)
        f.write(f'{timestamp} [{level}] {component}: Processing request {i} with data size {random.randint(100, 10000)} bytes\\n')
        if i % 1_000_000 == 0:
            print(f'Progress: {i:,} log entries written')
print('✅ Created massive_log.log (~1GB)')
"
```

## 🧪 Testing Without Large Files

If you don't want to create large files, the streaming tests will:

1. **Skip gracefully** if files are missing (won't fail the test suite)
2. **Generate small temporary files** for basic functionality testing
3. **Log warnings** about missing large files for performance testing

## 📊 Performance Expectations

With proper large files, you should see:

- **Streaming progress reports** every 1000 chunks
- **Memory usage** staying constant (not growing with file size)
- **Checkpoint creation** for resume capabilities
- **Multi-file distributed processing** across CPU cores
- **Throughput metrics** (MB/s, chunks/s)

## 🎯 File Size Guidelines

| File Type | Minimum Size | Recommended Size | Purpose |
|-----------|-------------|------------------|---------|
| Text      | 100MB       | 5GB+            | Basic streaming |
| Binary    | 500MB       | 10GB+           | Memory mapping |
| CSV       | 200MB       | 2GB+            | Structured data |
| Logs      | 50MB        | 1GB+            | Pattern processing |

## 🛠️ Advanced Features Tested

The streaming system will test:

- ✅ **Progress Reporting**: Real-time status updates
- ✅ **Checkpointing**: Resume interrupted processing
- ✅ **Distributed Processing**: Multi-file parallel processing
- ✅ **Memory Efficiency**: Constant memory usage regardless of file size
- ✅ **Error Recovery**: Graceful handling of corrupted blocks
- ✅ **Adaptive Block Sizing**: Dynamic optimization based on content

## 💡 Tips

1. **Use SSDs** for better I/O performance with large files
2. **Monitor system resources** during tests to verify streaming efficiency
3. **Check generated files** with `head`, `tail`, and `wc -l` commands
4. **Clean up** large files after testing if disk space is limited

## 🤝 Contributing

When adding new streaming tests:
1. Always make tests **gracefully skip** if large files are missing
2. Document the **expected file format** in test docstrings
3. Add entries to this README for **new required files**
4. Test both **with and without** large files present
