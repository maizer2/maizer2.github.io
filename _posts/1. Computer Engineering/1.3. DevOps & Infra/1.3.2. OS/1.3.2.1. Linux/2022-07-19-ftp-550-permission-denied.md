---
layout: post
categories: [1. Computer Engineering, 1.3. DevOps & Infra, 1.3.2. OS, 1.3.2.1. Linux]
title: "FTP error of 550 Permission denied error"
tags: [FTP, vsftpd]
---

### Situation

When I send DataSet file on windows to Ubuntu-server using FTP service.

### Error log

```cmd
...
200 PORT command successful. Consider using PASV.
550 Permission denied.
```

### Solution<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

In ubuntu, edit the vsftpd.conf

```ubuntu-server
sudo vi /etc/vsftpd.conf

...
write_enable=YES
...

:wq

sudo service vsftpd restart
```

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> Fixing the VSFTPD 550 Permission Denied Error, DevOperations, Write:2011.01.01, attach:2022.07.19, [https://whiteglass.tistory.com/12](https://whiteglass.tistory.com/12)