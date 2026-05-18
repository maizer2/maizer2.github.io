---
layout: post
categories: [1. Computer Engineering, 1.3. DevOps & Infra, 1.3.2. OS, 1.3.2.1. Linux]
title: "Docker host SSH-key share on ubuntu container"
tags: [1.2. Programming, 1.3.2.1. Linux, 1.3.2. OS, 1.3.3.1. Docker, 1.3.3. Container, 1.3.4. Network, 1.3. DevOps & Infra]
---

### Quick note

```
# Docker host (like window)

ssh-keygen

~

# Ssh personal key move to container
docker cp ../.ssh/[key_name] [container_name]:~/.ssh/[key_name]

# Ssh public key move to container
docker cp ../.ssh/[key_name.pub] [container_name]:~/.ssh/[key_name.pub]

# Attach the container
docker attach [container_name]

# Docker container (like ubuntu)

# Give rights the ssh personal key
chmod 600 ~/.ssg/[key_name]
```

---
### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> Sieg, Jan 3 2022, Setting up an SSH Key for your containers using docker compose, [https://denzehldacuyan.medium.com/setting-up-an-ssh-key-for-your-containers-using-docker-compose-d1aac4732c15](https://denzehldacuyan.medium.com/setting-up-an-ssh-key-for-your-containers-using-docker-compose-d1aac4732c15)