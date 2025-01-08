# fiftyone-getting-started

If the mongo process gets stuck

```

for prc in /proc/*/cmdline; { (printf "$prc "; cat -A "$prc") | sed 's/\^@/ /g;s|/proc/||;s|/cmdline||'; echo; } |grep mongo

```

then:

```
kill <number at the beginning of the returned line>
```