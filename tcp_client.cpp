// image_client_sim.cpp : ���ļ����� "main" ����������ִ�н��ڴ˴���ʼ��������
//

#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include<time.h>
#pragma comment(lib,"ws2_32.lib")
// CEVFPictureBox
/*
1.SYNC_HEADͷÿ������һ��
2.������������ΪREQ�����ؽ������ΪACK
3.width��heightΪͼ����
4.src_jpg_pic_lengthΪ���������ʱ�洢ԭʼͼ��jpg�ļ��ĳ���
5.mask_jpg_lengthΪ�洢���ؽ��ʱ��mask jpgͼ���ļ�����
6.result_jpg_lengthΪ�洢���ؽ��ʱ�ĺϳɽ����jpgͼ���ļ�����
7.��data��ʼ�����δ洢src_jpg_pic��mask_jpg��result_jpg���ļ�����
8.��������У�mask_jpg_length��result_jpg_length��������Ϊ0��Ҳ���Ǻ�����data��ֻ��src_jpg
9.�ڷ��ؽ�����У�src_jpg_pic_length��������Ϊ0��Ҳ���ǲ���Ҫ����ԭʼͼ���ļ�����
10.obj_area / obj_cir / obj_width / obj_height�ֱ�Ϊ���ؽ��ʱ���ϱ���Ŀ������������ܳ�������
*/
#define PIC_PROC_TCP_PORT (20010)
#define PIC_PROC_PACKET_SYNC_HEAD (0xa55aa55affffffffL)
#define PIC_PROC_PACKET_TYPE_REQ (0x1)
#define PIC_PROC_PAKCET_TYPE_ACK (0x2)

#pragma pack(1)
struct PIC_PROC_PACKET_S
{
	unsigned long long sync_head;
	unsigned long long pic_id;
	int pkt_type;
	int width;
	int height;
	float obj_width;
	float obj_height;
	float obj_area;
	float obj_cir;
	int src_jpg_pic_length;
	int mask_jpg_length;
	int result_jpg_length;
	char data[1];
};

int main(void)
{
	clock_t start, finish;
	start = clock();
	PIC_PROC_PACKET_S* ppps_send, * ppps_recv;
	SOCKET socket_fd;
	WSADATA wsaData;
	int filelen;
	FILE* fp;

	int ret, recv_len, recv_rest, recv_sum_len;
	ret = WSAStartup(MAKEWORD(2, 2), &wsaData);
	char* tmp;
	tmp = new char[50 * 1024 * 1024];
	memset(tmp, 0, sizeof(PIC_PROC_PACKET_S));
	ppps_send = (PIC_PROC_PACKET_S*)tmp;

	ppps_send->sync_head = PIC_PROC_PACKET_SYNC_HEAD;
	ppps_send->pkt_type = PIC_PROC_PACKET_TYPE_REQ;
	tmp = new char[50 * 1024 * 1024];
	ppps_recv = (PIC_PROC_PACKET_S*)tmp;
	fopen_s(&fp, "test.jpg", "rb+");
	filelen = fread(ppps_send->data, 1, 50 * 1024 * 1024, fp);
	ppps_send->src_jpg_pic_length = filelen;
	fclose(fp);
	ppps_recv->sync_head = PIC_PROC_PACKET_SYNC_HEAD;
	socket_fd = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	SOCKADDR_IN server_addr;
	server_addr.sin_family = AF_INET; //InternetЭ��
	server_addr.sin_port = htons(PIC_PROC_TCP_PORT);

	inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);
	if (0 > connect(socket_fd, (SOCKADDR*)&server_addr, sizeof(SOCKADDR))) //����ʧ���򷵻�SOCKET_ERROR
	{
		int errorCode = ::WSAGetLastError();
		char errstr[256];
		sprintf_s(errstr, "ͼ�������������ʧ��:%d", errorCode);
		::MessageBoxA(NULL, errstr, "����ʧ��", MB_OK);
	}

	ppps_send->sync_head = PIC_PROC_PACKET_SYNC_HEAD;
	ppps_send->pkt_type = PIC_PROC_PACKET_TYPE_REQ;

	ppps_send->width = 2592;
	ppps_send->height = 1728;
	send(socket_fd, (char*)ppps_send, sizeof(PIC_PROC_PACKET_S) - 1 + ppps_send->src_jpg_pic_length, 0);
	printf("header_size=%d\n", sizeof(PIC_PROC_PACKET_S) - 1);
	recv_len = recv(socket_fd, (char*)ppps_recv, sizeof(PIC_PROC_PACKET_S) - 1, 0);
	if (recv_len != (sizeof(PIC_PROC_PACKET_S) - 1))
	{
		printf("recv_head_len err=%d(recv_len=%d)\n", WSAGetLastError(), recv_len);
		exit(0);
	}
	printf("obj width is %f, obj height is %f, area is %f, cir is %f.\n", ppps_recv->obj_width, ppps_recv->obj_height, ppps_recv->obj_area, ppps_recv->obj_cir);
	printf("result file length is %d Bytes, mask file length is %d Bytes.\n", ppps_recv->result_jpg_length, ppps_recv->mask_jpg_length);
	recv_rest = ppps_recv->mask_jpg_length + ppps_recv->result_jpg_length;
	recv_sum_len = 0;
	while (recv_rest > 0)
	{
		//printf("Before:rest data=%d, recv_len=%d\n", recv_rest, recv_len);
		recv_len = recv(socket_fd, ((char*)ppps_recv) + (sizeof(PIC_PROC_PACKET_S) - 1) + (recv_sum_len), recv_rest, 0);

		if (recv_len > 0)
		{
			recv_rest -= recv_len;
			recv_sum_len += recv_len;
			//printf("After:rest data=%d, recv_sum_len=%d\n", recv_rest, recv_sum_len);
		}
		else
		{
			printf("recv err ret=%d err=%d\n", recv_len, WSAGetLastError());
		}
	}
	printf("recv %d Bytes data.\n", recv_sum_len + sizeof(PIC_PROC_PACKET_S) - 1);

	printf("save to result.jpg and mask.jpg.\n");
	fopen_s(&fp, "result.jpg", "wb+");
	fwrite(ppps_recv->data + ppps_recv->mask_jpg_length, 1, ppps_recv->result_jpg_length, fp);
	fclose(fp);
	fopen_s(&fp, "mask.jpg", "wb+");
	fwrite(ppps_recv->data, 1, ppps_recv->mask_jpg_length, fp);
	fclose(fp);

	closesocket(socket_fd);

	tmp = (char*)ppps_send;
	delete tmp;
	tmp = (char*)ppps_recv;
	delete tmp;
	finish = clock();
	double duration = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("%f seconds\n", duration);
	return 0;
}