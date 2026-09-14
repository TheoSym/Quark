umask 002
# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
[ -z "$PS1" ] && return

# don't put duplicate lines in the history. See bash(1) for more options
# ... or force ignoredups and ignorespace
HISTCONTROL=ignoredups:ignorespace

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "$debian_chroot" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color) color_prompt=yes;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
#force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
	# We have color support; assume it's compliant with Ecma-48
	# (ISO/IEC-6429). (Lack of such support is extremely rare, and such
	# a case would tend to support setf rather than setaf.)
	color_prompt=yes
    else
	color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm*|rxvt*)
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
*)
    ;;
esac

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
#if [ -f /etc/bash_completion ] && ! shopt -oq posix; then
#    . /etc/bash_completion
#fi

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/opt/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/opt/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

VAST_CONTAINERLABEL="$(cat ~/.vast_containerlabel)"
PS1="\[\e]0;\u@$VAST_CONTAINERLABEL: \w\a\]\[\e[01;34m\]\u\[\e[m\e[01m\]@\[\e[01;36m\]$VAST_CONTAINERLABEL\[\e[m\e[01m\]:\[\e[01;37m\]\w\$\[\e[m\] " ; if [ ! -e "$HOME/.no_auto_tmux" ] && [[ -z "$TMUX" ]] && [ "$SSH_CONNECTION" != "" ] && [ "$TMUX_STARTED" = "" ]; then tmux attach-session -t ssh_tmux || tmux new-session -s ssh_tmux; exit; elif ! [[ -z "$TMUX" ]]; then echo 'Welcome to your vast.ai container! This session is running in `tmux`.'; echo 'To disconnect without closing your processes, press ctrl+b, release, then d.'; echo 'To disable auto-tmux, run `touch ~/.no_auto_tmux` and reconnect. See also https://tmuxcheatsheet.com/'; fi; 
DIRECT_PORT_START=0
DIRECT_PORT_END=0
VAST_CONTAINERLABEL=C.50681041
### Entrypoint setup ###
cd ${WORKSPACE}
set -a
. /etc/environment
[[ -f "${WORKSPACE}/.env" ]] && . "${WORKSPACE}/.env"
set +a
[[ -f /opt/nvm/nvm.sh ]] && . /opt/nvm/nvm.sh
[[ ${CONDA_SHLVL:-0} = 0 && -f /venv/${ACTIVE_VENV:-main}/bin/activate ]] && . /venv/${ACTIVE_VENV:-main}/bin/activate
[[ -f /.provisioning ]] && echo -e "\e[91m>>>\e[0m Instance provisioning is not yet complete.\n\e[91m>>>\e[0m Required software may not be ready.\n\e[91m>>>\e[0m See /var/log/portal/provisioning.log or the Instance Portal web app for progress updates\n\n"
### End entrypoint setup ###
