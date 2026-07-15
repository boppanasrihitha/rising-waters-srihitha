document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("form");
    if (form) {
        form.addEventListener("submit", function (e) {
            const inputs = form.querySelectorAll("input[required]");
            let valid = true;
            inputs.forEach(input => {
                if (!input.value) {
                    valid = false;
                }
            });
            if (!valid) {
                e.preventDefault();
                alert("Please fill all fields before submitting.");
            }
        });
    }
});
