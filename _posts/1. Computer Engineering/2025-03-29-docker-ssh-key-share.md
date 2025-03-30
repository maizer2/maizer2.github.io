---
layout: post
title: "Docker host SSH-key share on ubuntu container"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.4. OS, 1.4.1. Linux, 1.5. Container, 1.5.1. Docker, 1.7. Network]
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