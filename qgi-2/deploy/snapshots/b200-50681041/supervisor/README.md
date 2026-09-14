# Supervisor wrapper scripts

Always use a wrapper script so the startup can be managed. e.g. 

- Skip launching a WebUI if the process is not in the `PORTAL_CONFIG` or `/etc/portal.yaml`

`PROC_NAME` is used in the default scripts and is set by the supervisor config files located in /etc/supervisor/conf.d/