/*******************************************************************
*
*  Modul zur Ausgabe von 80-Zeichen-Saetzen in eine Datei, 
*  mit den folgenden Funktionen:
*	buf_init()	initialisiert den Ausgabepuffer,
*	buf_putchar(char)
*			schreibt ein Zeichen in den Ausgabepuffer
*	buf_dump(FILE *)
*			schreibt den Ausgabepuffer in die Ausgabedatei
*	buf_free()	gibt den Ausgabepuffer wieder frei,
*			(wird bei buf_init() automatisch ausgefuehrt)
*
*  Zu dem Modul gehoeren die Dateien:
*	buf.c		Sourcecode
*	buf.h		Includes fuer externe Benutzer
*	buf_intern.h	Includes modulintern
*
*  A. Greve	V1.01	30.4.98
*  dstoll   V1.02   16.6.98
*
*******************************************************************
*/
#include <stdio.h>
#include <stdlib.h>
#include "gse_header.h"
#include "gse_types.h"
#include "buf.h"
#include "buf_intern.h"
/*
* Globale Variablen fuer die 80-Zeichen-Saetze
*/
static struct BufLine *bufBegin, *bufAct;
/*
******************************************************************
*
* Funktion: buf_init
*	Initialisierung der Speicherverwaltung muss unbedingt
*	als erstes aufgerufen werden
* Parameter: keine
*
******************************************************************
*/
int buf_init()
{
/*
	extern struct BufLine *bufBegin, *bufAct;
*/
/*
 *	dstoll: change "buf_free" to "buf_free()"
 */
	if (bufBegin != NULL) buf_free();
	bufBegin = buf_getNewLine();
	bufAct = bufBegin;
/*
 *	dstoll add:
 */
	return 0;
}
/*****************************************************************
*
* Funktion: buf_putchar
*	schreibt ein Zeichen in den Ausgabepuffer. Reicht dabei die
*	aktuelle Zeile nicht aus, so wird eine neue angefordert.
* Parameter: char C	zu schreibendes Zeichen
*
******************************************************************
*/
int buf_putchar(char C)
{
/*
	extern struct BufLine *bufAct;
*/
	switch (buf_putCharToLine(C,bufAct))
	{
	/* Einfuegen in vorhandene Zeile */
	case 0: return(0);
	/* neue Zeile notwendig, da aktuelle Zeile voll */
	case 1:
		bufAct->nextLine = buf_getNewLine();
		bufAct = bufAct->nextLine;
		return (buf_putchar(C));
	/* Pufferzeile fehlt */
	default:
		buf_err(FATAL,"buf_putchar","kein Schreibpuffer vohanden ----- Initialisierung fehlt wahrscheinlich");
		return(-1);
	}
}
/*****************************************************************
*
* Funktion: buf_dump
*	schreibt den gesamten Puffer in die Ausgabedatei.
*	Die Ausgabedatei muss dazu natuerlich geoeffnet sein.
* Parameter: FILE *fp	Pointer auf Ausgabedatei
*
******************************************************************
*/
int buf_dump(FILE *fp)
{
/*
	extern struct BufLine *bufBegin, *bufAct;
*/
	struct BufLine *bufDummy;
	/* kein Puffer vorhanden, also nichts zu tun */
	if (bufBegin == NULL)
	{
		buf_err(WARNING,"buf_dump","Buffer nicht initialisiert");
		return(1);
	}
	/* Ausgabe der Stichwortzeile */
	fprintf(fp,"DAT2\n");
	/* Ausgabe bis zur vorletzten Zeile */
	bufDummy = bufBegin;
	while (bufDummy->nextLine != NULL)
	{
		fprintf(fp,"%s\n",bufDummy->line);
		bufDummy = bufDummy->nextLine;
	}
	/* Ausgabe der letzten Zeile mit Blanks aufgefuellt */
/*
 *	dstoll: add next line
 */
	bufDummy->line[bufDummy->position] = '\0';
/*
 *	Die letzte Zeile muss korrekt terminiert sein, sonst kann es passieren,
 *	dass fprintf zu viel schreibt!
 */
	fprintf(fp,"%-80s\n",bufDummy->line);
	/* Wenn am Ende der letzten Zeile nich mindestens 2 Blanks sind,
	Ausgabe einer Leerzeile */
	if (bufDummy->position > (MAXPOS-1))
	{
		fprintf(fp,"%80s\n","");
	}
	return(0);
}
/*****************************************************************
*
* Funktion: buf_free
*	gibt den Pufferspeicher wieder frei.
* Parameter: keine
*
******************************************************************
*/
int buf_free()
{
	extern struct BufLine *bufBegin, *bufAct;
	bufAct = bufBegin;
	while (bufBegin != NULL)
	{
		bufBegin = bufBegin->nextLine;
		free(bufAct);
		bufAct=bufBegin;
	}
	return(0);
}
/*******************************************************************
*
*		Hilfsfunktionen (intern)
*
*******************************************************************/

/****************************************************************
*
* Funktion: buf_getNewLine
*	erzeugt eine neue Zeile und gibt einen Zeiger auf diese
*	zurueck
*
*****************************************************************
*/
struct BufLine *buf_getNewLine()
{
	struct BufLine *bufLine;
	bufLine = (struct BufLine *) malloc (sizeof(struct BufLine));
	bufLine->position = 0;
	bufLine->nextLine = (struct BufLine *)NULL;
	return(bufLine);
}
/****************************************************************
*
* Funktion: buf_putCharToLine
*	schreibt ein Zeichen in die interne BuFLine-Struktur
*	und setzt dabei auch den entprechenden Zeilenzeiger
*	in der Struktur.
*
*****************************************************************
*/
int buf_putCharToLine(char C, struct BufLine *bufLine)
{
	if (bufLine == NULL) {
		/*@@@*/printf("buf_putCharToLine called with NULL\n");
		return(-1);
	}
	if (bufLine->position > MAXPOS)
	{
		bufLine->line[MAXPOS+1] = '\0';
		return(1);
	}
	bufLine->line[bufLine->position] = C;
	bufLine->position++;
	return(0);
}
/****************************************************************
*
* Funktion: buf_err
*	Fehlerbehandlung
*
*****************************************************************
*/
void buf_err(int mode, char *func_name, char *message)
{
	switch (mode)
	{
	case FATAL:
		printf ("Fatal Error in Funktion %s:\n%s\n",func_name,message);
		exit(0);
	case WARNING:
		printf ("Warning in Funktion %s:\n%s\n",func_name,message);
		break;
	default:
		printf ("whoooops\n");
		break;
	}
}

int compress_6b(int32_t *data, int n_of_samples) {
    return compress_6b_buffer(data, n_of_samples, &buf_putchar);
}

char * get_line_83(char * cbuf, void * fop) {
    return fgets (cbuf,83,fop);
}

int decomp_6b(FILE *fop, int n_of_samples, int32_t *dta) {
    return decomp_6b_buffer (n_of_samples, dta, &get_line_83,(void *) fop);
}

/******************************************************************
*
*	Testprogramm
*
******************************************************************/
#ifdef MAIN
main()
{
	int i,j;
	FILE *fp;
	fp = fopen("testausgabe.txt","w");
	buf_init();
	for(j=33;j<126;j++)
	for (i=0;i<=MAXPOS;i++)
	{
		buf_putchar((char)j);
	}
	buf_putchar('E');
	buf_putchar('n');
	buf_putchar('D');
	buf_putchar('E');

	buf_dump(fp);
	fclose(fp);
	buf_free();
}
#endif /* MAIN */
