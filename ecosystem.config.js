module.exports = {
  apps: [
    {
      name: 'au-scoping',
      script: 'app.py',
      interpreter: 'python3',
      cwd: '/home/ubuntu/AU-Scoping',
      env: {
        FLASK_ENV: 'production',
        PORT: 5007,
        PYTHONPATH: '.'
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      error_file: './logs/err.log',
      out_file: './logs/out.log',
      log_file: './logs/combined.log',
      time: true
    }
  ]
};