---
layout: post
title: "FTP error of 550 Permission denied error"
categories: [1. Computer Engineering]
tags: [1.4. OS, 1.4.1. Linux, 1.7. Network]
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