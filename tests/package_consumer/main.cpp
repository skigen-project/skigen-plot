#include <skigen/plot/core.h>

int main()
{
    Skigen::Plot::BoundingBox2D bounds;
    bounds.min = {0.0f, 0.0f};
    bounds.max = {1.0f, 1.0f};
    return bounds.expanded(0.1f).width() > bounds.width() ? 0 : 1;
}
