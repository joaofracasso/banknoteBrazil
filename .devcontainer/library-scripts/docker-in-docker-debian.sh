#!/usr/bin/env bash
#-------------------------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See https://go.microsoft.com/fwlink/?linkid=2090316 for license information.
#-------------------------------------------------------------------------------------------------------------
#
# Docs: https://github.com/microsoft/vscode-dev-containers/blob/master/script-library/docs/docker-in-docker.md
# Maintainer: The VS Code and Codespaces Teams
#
# Syntax: ./docker-in-docker-debian.sh [enable non-root docker access flag] [non-root user] [use moby]

set -e

if [ "$(id -u)" -ne 0 ]; then
    echo -e 'Script must be run as root. Use sudo, su, or add "USER root" to your Dockerfile before running this script.'
    exit 1
fi

echo "Finished installing docker / moby"

# Install Docker Compose if not already installed 
if type docker-compose > /dev/null 2>&1; then
    echo "Docker Compose already installed."
else
    LATEST_COMPOSE_VERSION=$(curl -sSL "https://api.github.com/repos/docker/compose/releases/latest" | grep -o -P '(?<="tag_name": ").+(?=")')
    curl -sSL "https://github.com/docker/compose/releases/download/${LATEST_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# If init file already exists, exit
if [ -f "/usr/local/share/docker-init.sh" ]; then
    echo "/usr/local/share/docker-init.sh already exists, so exiting."
    exit 0
fi
echo "docker-init doesnt exist..."



tee /usr/local/share/docker-init.sh > /dev/null \
<< 'EOF' 
#!/usr/bin/env bash
#-------------------------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See https://go.microsoft.com/fwlink/?linkid=2090316 for license information.
#-------------------------------------------------------------------------------------------------------------
sudoIf()
{
    if [ "$(id -u)" -ne 0 ]; then
        sudo "$@"
    else
        "$@"
    fi
}
# explicitly remove dockerd and containerd PID file to ensure that it can start properly if it was stopped uncleanly
# ie: docker kill <ID>
sudoIf find /run /var/run -iname 'docker*.pid' -delete || :
sudoIf find /run /var/run -iname 'container*.pid' -delete || :
set -e
## Dind wrapper script from docker team
# Maintained: https://github.com/moby/moby/blob/master/hack/dind
export container=docker
if [ -d /sys/kernel/security ] && ! sudoIf mountpoint -q /sys/kernel/security; then
	sudoIf mount -t securityfs none /sys/kernel/security || {
		echo >&2 'Could not mount /sys/kernel/security.'
		echo >&2 'AppArmor detection and --privileged mode might break.'
	}
fi
# Mount /tmp (conditionally)
if ! sudoIf mountpoint -q /tmp; then
	sudoIf mount -t tmpfs none /tmp
fi
# cgroup v2: enable nesting
if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
	# move the init process (PID 1) from the root group to the /init group,
	# otherwise writing subtree_control fails with EBUSY.
	sudoIf mkdir -p /sys/fs/cgroup/init
	sudoIf echo 1 > /sys/fs/cgroup/init/cgroup.procs
	# enable controllers
	sudoIf sed -e 's/ / +/g' -e 's/^/+/' < /sys/fs/cgroup/cgroup.controllers \
		> /sys/fs/cgroup/cgroup.subtree_control
fi
## Dind wrapper over.
# Handle DNS
set +e
cat /etc/resolv.conf | grep -i 'internal.cloudapp.net'
if [ $? -eq 0 ]
then
  echo "Setting dockerd Azure DNS."
  CUSTOMDNS="--dns 168.63.129.16"
else
  echo "Not setting dockerd DNS manually."
  CUSTOMDNS=""
fi
set -e
# Start docker/moby engine
( sudoIf dockerd $CUSTOMDNS > /tmp/dockerd.log 2>&1 ) &
set +e
# Execute whatever commands were passed in (if any). This allows us 
# to set this script to ENTRYPOINT while still executing the default CMD.
exec "$@"
EOF

chmod +x /usr/local/share/docker-init.sh