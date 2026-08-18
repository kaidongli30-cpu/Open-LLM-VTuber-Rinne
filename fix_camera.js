// 强制所有 getUserMedia 调用都使用后置摄像头
(function() {
    const originalGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = function(constraints) {
        if (constraints.video) {
            constraints.video = { facingMode: { exact: "environment" } };
        }
        return originalGetUserMedia(constraints);
    };
})();