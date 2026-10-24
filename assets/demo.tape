# Records assets/demo.gif. Run from the repo root after `make && make model`:
#   vhs assets/demo.tape

Output assets/demo.gif

Set Shell "bash"
Set FontSize 15
Set Width 1180
Set Height 560
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms

Hide
Type "export PS1='$ ' && clear"
Enter
Show

Type "python3 scripts/demo.py --kill-at 80"
Sleep 400ms
Enter
Sleep 9s
