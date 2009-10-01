#define MAXPOS 79
#define FATAL 1000
#define WARNING 1001
struct BufLine
{
	char line[MAXPOS+2];
	int position;
	struct BufLine *nextLine;
};
struct BufLine *buf_getNewLine();
int buf_putCharToLine(char,struct BufLine *);
void buf_err(int, char *, char *);
