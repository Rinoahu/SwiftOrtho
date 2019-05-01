from cffi import FFI
ffi = FFI()
ffi.cdef("""
int len(char *s[]);
""")
C = ffi.verify(r"""
#include<stdio.h>
#include<string.h>

int len(char *s[]){
printf("%s %s xxx\n", s[0], (s[1]));
printf("%d %d yyy\n", strlen(s[0]), strlen(s[1]));
return 1;
}
""")

x = ffi.new('char *p[2]')
x[0] = ffi.new('char []', 'abcd')
x[1] = ffi.new('char []', 'abcdef')

C.len(x)

