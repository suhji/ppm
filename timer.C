#include <stdio.h>   
#include <string.h>   
 #include <stdlib.h> // exit()

#define MAX 255   // char 형 변수의 크기를 지정할 변수
#include <time.h> // time()



 void main(int argc, char *argv[])
{
     int j=1;
    clock_t start, finish;
    double duration=0.0;

	unsigned int i, begin_word = 0, count = 0;
    char fn[MAX], buffer[MAX];
	char String[MAX];  // 한 줄씩 읽어 올 문자열을 저장할 공간

	FILE *fp;
	fopen_s(&fp,"test.txt","r");
	for(;;)
	{fgets(fn,255,fp);
	if(feof(fp))
		break;
	printf("%s",fn);}

    if (argc > 1)
        strcpy(fn, argv[1]);
    else
        strcpy(fn, "test.txt");// test.txt로 저장된 파일이 본체에 있어야한다는 번거로움.


    if ((fp = fopen(fn, "r")) == NULL) {
        printf("Cannot open the file: %s\n", fn);
        exit(0);
    }
	
	

    while (!feof(fp)) {
	
		if (fgets(buffer, MAX, fp) == NULL) break;// 파일을 끝까지 읽었는지 체크
                   
         
        for (i = 0; i < strlen(buffer); i++) {
            if (buffer[i] == ' ' || buffer[i] == '\n' || buffer[i] == '\t') {
                if (begin_word) {
                    count++;
                    begin_word = 0;
                }
            }
            else {
                if (!begin_word)
                    begin_word = 1;
            }
        }
    }

    fclose(fp);

    printf("The number of words in \"%s\" is %d.\n", fn, count);
  //클릭할지 보이스인식할지에 따라 다름
	while(j){
        printf("\nIf you want to start Timer, Click 'Enter'>");
        getchar();
        start = clock();
        
        printf("\nstarting... If you want to finish Timer, Click 'Enter' again>");
        getchar();
        finish = clock();

        duration = (double)(finish-start)/CLOCKS_PER_SEC;
        printf("\n\n\t%f sec\n", duration);
		
		
			
	if(duration<count*0.375){
		printf("TOO FAST.\n");}
	
	if(duration>count*0.6){
		printf("TOO SLOW.\n");}

        printf("\nTo restart Click 'Enter', To finish Click '0' and 'Enter'>");
        j = getchar();
        
        if(j==48){
            j=0;
        }else{
            fflush(stdin);
            printf("\n========\n");
        }
    }
   }