import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  transcript = '';
  note = '';
  selectedFile?: File;

  constructor(private http: HttpClient) {}

  onFileSelected(event: any) {
    const file: File = event.target.files[0];
    if (file) {
      this.selectedFile = file;
    }
  }

  upload() {
    if (!this.selectedFile) { return; }
    const formData = new FormData();
    formData.append('file', this.selectedFile);
    formData.append('model', 'vosk_small');
    this.http.post<{transcript: string}>('/api/transcribe', formData)
      .subscribe(res => this.transcript = res.transcript);
  }

  generateNote() {
    this.http.post<{note: string}>('/api/notes', { transcript: this.transcript })
      .subscribe(res => this.note = res.note);
  }
}
